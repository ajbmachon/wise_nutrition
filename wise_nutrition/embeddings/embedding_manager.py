"""
Embedding manager module.
"""
from typing import List, Optional, Any, Dict

import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Weaviate
from langchain_unstructured import UnstructuredLoader
from wise_nutrition.utils.config import Config

# Example of a document chunk for recipe 
# {
#   "title": "Piima Starter Culture",
#   "section": "Cultured Dairy",
#   "text": "Full original recipe including ingredients, detailed preparation steps, and traditional or scientific context.",
#   "nutrients": ["Probiotics", "Vitamin A", "Vitamin K2"],
#   "chunk_id": "recipe_piima_starter_culture"
# }
#
# Example of a document chunk for Nutrition Theory
# {
#   "section": "Fermentation and Lactic Acid Bacteria",
#   "quote": "Traditional societies often allowed grains, milk products, and vegetables to ferment via lacto-fermentation...",
#   "summary": "Describes how traditional diets used fermentation to preserve food and improve nutrient availability.",
#   "tags": ["fermentation", "gut health", "lactic acid"],
#   "source": "Nourishing Traditions, p. 35",
#   "chunk_id": "theory_fermentation_lactic"
# } 

class EmbeddingManager:
    """
    Manage document embeddings and storage in Weaviate.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        collection_name: Optional[str] = None,
        vector_store: Optional[Weaviate] = None
    ):
        """
        Initialize the embedding manager.
        
        Args:
            config: Configuration object containing API keys and URLs
            collection_name: Name of the collection in Weaviate
        """
        self._config = config or Config()
        self._collection_name = collection_name or self._config.weaviate_collection_name
        
        # Initialize Weaviate client
        self._client = weaviate.Client(
            url=self._config.weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=self._config.weaviate_api_key),
        )
        
        # Initialize OpenAI embeddings
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=self._config.openai_api_key
        )
        
        # Initialize Weaviate vector store instance
        self._vector_store = None
    
    def create_collection(self) -> None:
        """
        Create or reset the Weaviate collection.
        
        Deletes the collection if it exists, then creates a new one.
        """
        # Check if collection exists and delete it
        if self._client.schema.exists(self._collection_name):
            self._client.schema.delete_class(self._collection_name)
            print(f"Deleted existing collection: {self._collection_name}")
        
        # Define the schema for the collection
        schema = {
            "classes": [
                {
                    "class": self._collection_name,
                    "description": "Nutrition documents including recipes and theory",
                    "vectorizer": "text2vec-openai",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "model": "ada",
                            "modelVersion": "002",
                            "type": "text"
                        }
                    },
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                            "description": "The main text content"
                        },
                        {
                            "name": "title",
                            "dataType": ["text"],
                            "description": "The title of the document"
                        },
                        {
                            "name": "section",
                            "dataType": ["text"],
                            "description": "The section the document belongs to"
                        },
                        {
                            "name": "nutrients",
                            "dataType": ["text[]"],
                            "description": "List of nutrients in the recipe"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["text"],
                            "description": "Unique identifier for the document chunk"
                        },
                        {
                            "name": "type",
                            "dataType": ["text"],
                            "description": "Type of document (recipe, theory, etc.)"
                        }
                    ]
                }
            ]
        }
        
        # Create the collection
        self._client.schema.create(schema)
        print(f"Created collection: {self._collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the Weaviate collection.
        
        Args:
            documents: List of documents to add
        """
        # Initialize vector store if not already done
        self._initialize_vector_store() 
        
        # Add documents to Weaviate
        self._vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to Weaviate collection '{self._collection_name}'")
    
    def _initialize_vector_store(self, vector_store: Optional[Weaviate] = None) -> None:
        """
        Initialize the vector store.
        """
        if not self._vector_store:
            if not vector_store:
                vector_store = Weaviate(
                    client=self._client,
                    index_name=self._collection_name,
                    text_key="text",
                    embedding=self._embeddings,
                    by_text=False
                )
            else:
                self._vector_store = vector_store(
                    client=self._client,
                    index_name=self._collection_name,
                    text_key="text",
                    embedding=self._embeddings,
                    by_text=False
                )
        return self._vector_store
            
            
    def get_retriever(self, k: int = 4) -> Any:
        """
        Get a retriever for the Weaviate collection.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            A retriever instance
        """
        self._initialize_vector_store()
        
        return self._vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents based on a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        retriever = self.get_retriever(k=k)
        return retriever.get_relevant_documents(query)
    
    def document_exists(self, chunk_id: str) -> bool:
        """
        Check if a document with the given chunk_id exists in the collection.
        
        Args:
            chunk_id: The chunk_id to check
            
        Returns:
            True if the document exists, False otherwise
        """
        result = (
            self._client.query
            .get(self._collection_name, ["chunk_id"])
            .with_where({
                "path": ["chunk_id"],
                "operator": "Equal",
                "valueText": chunk_id
            })
            .do()
        )
        
        # Check if any results were returned
        return bool(result["data"]["Get"][self._collection_name]) 