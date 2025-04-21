"""Vector retrieval module for Wise Nutrition."""

from typing import List, Optional, Protocol, Dict, Any, Union
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from wise_nutrition.models import (
    NutrientInfo,
    Recipe,
    NutritionTheory,
    RetrievedContext,
)

# Set up logging
logging = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Vector store protocol defining required methods for retrieval."""
    
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
        ...
        
    def ingest_data(self, data_dir: Optional[Path] = None) -> None:
        """Ingest data from files into the vector store.
        
        Args:
            data_dir: Directory containing data files
        """
        ...


class SimpleFileVectorStore:
    """A simple file-based vector store for demonstration purposes.
    
    This implementation doesn't use actual vector embeddings, but performs
    basic keyword matching for demonstration. In production, this would be
    replaced with Supabase pgvector or another vector database.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize with path to data directory.
        
        Args:
            data_dir: Path to directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.vitamins_data = self._load_json(self.data_dir / "samples" / "vitamins.json")
        self.theory_data = self._load_json(self.data_dir / "samples" / "theory.json")
        self.recipes_data = self._load_json(self.data_dir / "samples" / "recipes.json")

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON data from file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data as a list of dictionaries
        """
        if not file_path.exists():
            return []
        
        with open(file_path, "r") as f:
            return json.load(f)
        
    async def search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for documents matching query using basic keyword matching.
        
        Args:
            query: The query text to search for
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of matching documents
        """
        keywords = query.lower().split()
        results = []
        
        # Search vitamins
        for item in self.vitamins_data:
            score = self._calculate_match_score(item["text"], keywords)
            if score > 0:
                results.append({
                    "item": item,
                    "score": score,
                    "type": "nutrient_info"
                })
        
        # Search theory
        for item in self.theory_data:
            # Combine section, quote and summary for matching
            combined_text = f"{item.get('section', '')} {item.get('quote', '')} {item.get('summary', '')}"
            score = self._calculate_match_score(combined_text, keywords)
            if score > 0:
                results.append({
                    "item": item,
                    "score": score,
                    "type": "theory"
                })
        
        # Search recipes
        for item in self.recipes_data:
            # Combine name, description, and nutrition benefits
            combined_text = (
                f"{item.get('name', '')} {item.get('description', '')} "
                f"{' '.join(item.get('nutrition_benefits', []))} "
                f"{' '.join(item.get('tags', []))}"
            )
            score = self._calculate_match_score(combined_text, keywords)
            if score > 0:
                results.append({
                    "item": item,
                    "score": score,
                    "type": "recipe"
                })
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _calculate_match_score(self, text: str, keywords: List[str]) -> float:
        """Calculate a simple match score based on keyword presence.
        
        Args:
            text: Text to search in
            keywords: Keywords to search for
            
        Returns:
            Match score (higher is better)
        """
        if not text or not keywords:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        
        for keyword in keywords:
            if keyword in text_lower:
                score += 1.0
                # Bonus for exact phrase matches
                if len(keyword) > 3 and f" {keyword} " in f" {text_lower} ":
                    score += 0.5
        
        return score


class RetrievalService:
    """Service for retrieving relevant information from vector storage."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize with a vector store.
        
        Args:
            vector_store: Vector store implementation
        """
        self.vector_store = vector_store
        
    def set_vector_store(self, vector_store: VectorStore) -> None:
        """Set a new vector store implementation.
        
        Args:
            vector_store: Vector store implementation
        """
        self.vector_store = vector_store
    
    async def retrieve_context(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievedContext:
        """Retrieve relevant context for the query.
        
        Args:
            query: User query
            limit: Maximum number of results
            filters: Optional filters
            
        Returns:
            Retrieved context with nutrition info, recipes, and theory
        """
        results = await self.vector_store.search(query, limit=limit, filters=filters)
        
        # Organize results by type
        nutrient_info = []
        recipes = []
        theory = []
        
        for result in results:
            item = result["item"]
            item_type = result["type"]
            
            if item_type == "nutrient_info":
                nutrient_info.append(NutrientInfo(**item))
            elif item_type == "recipe":
                recipes.append(Recipe(**item))
            elif item_type == "theory":
                theory.append(NutritionTheory(**item))
        
        return RetrievedContext(
            nutrition_info=nutrient_info,
            recipes=recipes,
            theory=theory
        )
