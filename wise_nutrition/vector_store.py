"""Vector store implementation using Supabase pgvector."""

from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

from openai import OpenAI

from wise_nutrition.config import Config
from wise_nutrition.db_init import get_vecs_client

# Set up logging
logger = logging.getLogger(__name__)


class SupabaseVectorStore:
    """Vector store implementation using Supabase pgvector and vecs client.
    
    This class manages vector collections for different types of data:
    - nutrient_info: Information about nutrients (vitamins, minerals, etc.)
    - theory: Nutrition theory and principles
    - recipes: Recipe data with nutritional benefits
    """
    
    def __init__(self):
        """Initialize the vector store with a connection to Supabase."""
        self.vecs_client = get_vecs_client()
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize collections
        self.nutrient_info = self._get_or_create_collection("nutrient_info")
        self.theory = self._get_or_create_collection("theory")
        self.recipes = self._get_or_create_collection("recipes")
    
    def _get_or_create_collection(self, name: str) -> Any:
        """Get or create a vector collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection object
        """
        return self.vecs_client.get_or_create_collection(
            name=name, 
            dimension=Config.EMBEDDING_DIMENSIONS
        )
    
    async def search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the vector store for similar documents.
        
        Args:
            query: The query text to search for
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of matching documents with similarity scores
        """
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)
        
        results = []
        
        # Search in nutrient_info collection
        nutrient_results = self._search_collection(
            self.nutrient_info, 
            query_embedding, 
            limit=limit, 
            filters=filters,
            item_type="nutrient_info"
        )
        results.extend(nutrient_results)
        
        # Search in theory collection
        theory_results = self._search_collection(
            self.theory, 
            query_embedding, 
            limit=limit, 
            filters=filters,
            item_type="theory"
        )
        results.extend(theory_results)
        
        # Search in recipes collection
        recipe_results = self._search_collection(
            self.recipes, 
            query_embedding, 
            limit=limit, 
            filters=filters,
            item_type="recipe"
        )
        results.extend(recipe_results)
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _search_collection(
        self, 
        collection: Any, 
        query_embedding: List[float], 
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        item_type: str = ""
    ) -> List[Dict[str, Any]]:
        """Search a specific collection.
        
        Args:
            collection: Vector collection to search
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters
            item_type: Type of item in the collection
            
        Returns:
            List of results with scores
        """
        try:
            # Convert filters to vecs format if provided
            filter_expr = None
            if filters:
                # Convert dictionary filters to vecs filter format
                filter_conditions = {}
                for k, v in filters.items():
                    filter_conditions[k] = {"$eq": v}
                filter_expr = filter_conditions
            
            # Query the collection using the simplified vecs API
            results = collection.query(
                query_vector=query_embedding,
                limit=limit,
                filters=filter_expr,
                include_metadata=True,
                include_value=True
            )
            
            # Format results
            formatted_results = []
            for result_id, result_distance, result_metadata in results:
                # Convert distance to similarity score (1 - distance)
                similarity_score = 1.0 - min(1.0, result_distance)
                
                formatted_results.append({
                    "item": result_metadata,
                    "score": similarity_score,
                    "type": item_type
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching collection {collection.name}: {str(e)}")
            return []
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def ingest_data(self, data_dir: Optional[Path] = None) -> None:
        """Ingest data from JSON files into vector collections.
        
        Args:
            data_dir: Directory containing data files. If None, uses Config.DATA_DIR
        """
        data_dir = data_dir or Config.DATA_DIR
        samples_dir = data_dir / "samples"
        
        logger.info(f"Ingesting data from {samples_dir}")
        
        # Ingest nutrient info
        self._ingest_nutrient_info(samples_dir / "vitamins.json")
        
        # Ingest theory
        self._ingest_theory(samples_dir / "theory.json")
        
        # Ingest recipes
        self._ingest_recipes(samples_dir / "recipes.json")
        
        # Create indexes for all collections for faster search
        self._create_indexes()
    
    def _ingest_nutrient_info(self, file_path: Path) -> None:
        """Ingest nutrient information from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        if not file_path.exists():
            logger.warning(f"Nutrient info file not found: {file_path}")
            return
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        vectors = []
        for item in data:
            # Generate a unique ID for the item
            item_id = f"nutrient_{item['metadata']['nutrient']}_{item['metadata']['category']}"
            
            # Generate embedding for the item text
            embedding = self._generate_embedding(item["text"])
            
            # Add to vectors list
            vectors.append((item_id, embedding, item))
        
        # Upsert vectors to the collection
        if vectors:
            self.nutrient_info.upsert(records=vectors)
            logger.info(f"Ingested {len(vectors)} nutrient info items")
    
    def _ingest_theory(self, file_path: Path) -> None:
        """Ingest nutrition theory from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        if not file_path.exists():
            logger.warning(f"Theory file not found: {file_path}")
            return
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        vectors = []
        for item in data:
            # Generate a unique ID for the item
            item_id = f"theory_{item['chunk_id']}"
            
            # Combine section, quote and summary for embedding
            combined_text = f"{item.get('section', '')} {item.get('quote', '')} {item.get('summary', '')}"
            
            # Generate embedding for the combined text
            embedding = self._generate_embedding(combined_text)
            
            # Add to vectors list
            vectors.append((item_id, embedding, item))
        
        # Upsert vectors to the collection
        if vectors:
            self.theory.upsert(records=vectors)
            logger.info(f"Ingested {len(vectors)} theory items")
    
    def _ingest_recipes(self, file_path: Path) -> None:
        """Ingest recipes from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        if not file_path.exists():
            logger.warning(f"Recipes file not found: {file_path}")
            return
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        vectors = []
        for item in data:
            # Generate a unique ID for the item
            item_id = f"recipe_{item['name'].lower().replace(' ', '_')}"
            
            # Combine name, description, and nutrition benefits for embedding
            combined_text = (
                f"{item.get('name', '')} {item.get('description', '')} "
                f"{' '.join(item.get('nutrition_benefits', []))} "
                f"{' '.join(item.get('tags', []))}"
            )
            
            # Generate embedding for the combined text
            embedding = self._generate_embedding(combined_text)
            
            # Add to vectors list
            vectors.append((item_id, embedding, item))
        
        # Upsert vectors to the collection
        if vectors:
            self.recipes.upsert(records=vectors)
            logger.info(f"Ingested {len(vectors)} recipe items")
    
    def _create_indexes(self) -> None:
        """Create indexes for all collections for faster search performance."""
        try:
            self.nutrient_info.create_index()
            logger.info("Created index for nutrient_info collection")
        except Exception as e:
            logger.error(f"Error creating index for nutrient_info: {str(e)}")
            
        try:
            self.theory.create_index()
            logger.info("Created index for theory collection")
        except Exception as e:
            logger.error(f"Error creating index for theory: {str(e)}")
            
        try:
            self.recipes.create_index()
            logger.info("Created index for recipes collection")
        except Exception as e:
            logger.error(f"Error creating index for recipes: {str(e)}")
