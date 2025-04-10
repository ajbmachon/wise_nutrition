"""
ChromaDB-based embedding manager module.
"""
import copy
import json
import os
import shutil
from typing import List, Optional, Any, Dict, Union

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from wise_nutrition.utils.config import Config


class ChromaEmbeddingManager:
    """
    Manage document embeddings and storage in ChromaDB.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        vector_store: Optional[Chroma] = None
    ):
        """
        Initialize the ChromaDB embedding manager.
        
        Args:
            config: Configuration object containing API keys
            collection_name: Name of the collection in ChromaDB
            persist_directory: Directory to persist ChromaDB data
            vector_store: Existing Chroma vector store instance
        """
        self._config = config or Config()
        self._collection_name = collection_name or "nutrition_collection"
        self._persist_directory = persist_directory or os.path.join(os.getcwd(), "chroma_db")
        
        # Initialize OpenAI embeddings
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=self._config.openai_api_key
        )
        
        # Initialize Chroma vector store instance
        self._vector_store = vector_store
    
    async def create_collection(self) -> None:
        """
        Create or reset the ChromaDB collection.
        
        Deletes the collection if it exists, then creates a new one.
        """
        # Check if collection directory exists and delete it
        if os.path.exists(self._persist_directory):
            shutil.rmtree(self._persist_directory)
            print(f"Deleted existing collection directory: {self._persist_directory}")
        
        # Create directory if it doesn't exist
        os.makedirs(self._persist_directory, exist_ok=True)
        
        # Initialize empty Chroma DB
        self._vector_store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory
        )
        
        # No need to call persist() as Chroma 0.4.x+ automatically persists
        print(f"Created collection: {self._collection_name}")
    
    def create_collection_sync(self) -> None:
        """
        Synchronous version of create_collection.
        """
        # Check if collection directory exists and delete it
        if os.path.exists(self._persist_directory):
            shutil.rmtree(self._persist_directory)
            print(f"Deleted existing collection directory: {self._persist_directory}")
        
        # Create directory if it doesn't exist
        os.makedirs(self._persist_directory, exist_ok=True)
        
        # Initialize empty Chroma DB
        self._vector_store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory
        )
        
        # No need to call persist() as Chroma 0.4.x+ automatically persists
        print(f"Created collection: {self._collection_name}")
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure compatibility with ChromaDB.
        
        Args:
            metadata: The metadata to sanitize
            
        Returns:
            Sanitized metadata with list values converted to strings
        """
        if not metadata:
            return metadata
            
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = value
                
        return sanitized
    
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the ChromaDB collection.
        
        Args:
            documents: List of documents to add
        """
        # Initialize vector store if not already done
        await self._initialize_vector_store()
        
        # Make a deep copy to avoid modifying original documents
        docs_to_add = copy.deepcopy(documents)
        
        # Transform the documents to include metadata
        for doc in docs_to_add:
            if not doc.metadata.get("chunk_id"):
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    doc.metadata["chunk_id"] = f"doc_{hash(doc.page_content)}"
            
            # Sanitize metadata to ensure compatibility with ChromaDB
            doc.metadata = self._sanitize_metadata(doc.metadata)
        
        # Add documents to ChromaDB
        self._vector_store.add_documents(docs_to_add)
        # No need to call persist() as Chroma 0.4.x+ automatically persists
        print(f"Added {len(documents)} documents to ChromaDB collection '{self._collection_name}'")
    
    def add_documents_sync(self, documents: List[Document]) -> None:
        """
        Synchronous version of add_documents.
        
        Args:
            documents: List of documents to add
        """
        # Initialize vector store if not already done
        self._initialize_vector_store_sync()
        
        # Make a deep copy to avoid modifying original documents
        docs_to_add = copy.deepcopy(documents)
        
        # Transform the documents to include metadata
        for doc in docs_to_add:
            if not doc.metadata.get("chunk_id"):
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    doc.metadata["chunk_id"] = f"doc_{hash(doc.page_content)}"
            
            # Sanitize metadata to ensure compatibility with ChromaDB
            doc.metadata = self._sanitize_metadata(doc.metadata)
        
        # Add documents to ChromaDB
        self._vector_store.add_documents(docs_to_add)
        # No need to call persist() as Chroma 0.4.x+ automatically persists
        print(f"Added {len(documents)} documents to ChromaDB collection '{self._collection_name}'")
    
    async def _initialize_vector_store(self) -> Chroma:
        """
        Initialize the vector store asynchronously.
        
        Returns:
            Initialized Chroma vector store
        """
        if not self._vector_store:
            # Check if the collection already exists
            if os.path.exists(self._persist_directory):
                self._vector_store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=self._persist_directory
                )
            else:
                # Create directory if it doesn't exist
                os.makedirs(self._persist_directory, exist_ok=True)
                
                # Initialize empty Chroma DB
                self._vector_store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=self._persist_directory
                )
                # No need to call persist() as Chroma 0.4.x+ automatically persists
        
        return self._vector_store
    
    def _initialize_vector_store_sync(self) -> Chroma:
        """
        Initialize the vector store synchronously.
        
        Returns:
            Initialized Chroma vector store
        """
        if not self._vector_store:
            # Check if the collection already exists
            if os.path.exists(self._persist_directory):
                self._vector_store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=self._persist_directory
                )
            else:
                # Create directory if it doesn't exist
                os.makedirs(self._persist_directory, exist_ok=True)
                
                # Initialize empty Chroma DB
                self._vector_store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=self._persist_directory
                )
                # No need to call persist() as Chroma 0.4.x+ automatically persists
        
        return self._vector_store
    
    async def get_retriever(self, k: int = 4) -> Any:
        """
        Get a retriever for the ChromaDB collection.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            A retriever instance
        """
        await self._initialize_vector_store()
        
        return self._vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    def get_retriever_sync(self, k: int = 4) -> Any:
        """
        Synchronous version of get_retriever.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            A retriever instance
        """
        self._initialize_vector_store_sync()
        
        return self._vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    async def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents based on a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        retriever = await self.get_retriever(k=k)
        return retriever.invoke(query)
    
    def search_similar_sync(self, query: str, k: int = 4) -> List[Document]:
        """
        Synchronous version of search_similar.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        retriever = self.get_retriever_sync(k=k)
        return retriever.invoke(query)
    
    async def document_exists(self, chunk_id: str) -> bool:
        """
        Check if a document with the given chunk_id exists in the collection.
        
        Args:
            chunk_id: The chunk_id to check
            
        Returns:
            True if the document exists, False otherwise
        """
        await self._initialize_vector_store()
        
        # Query for documents with matching chunk_id in metadata
        results = self._vector_store.get(
            where={"chunk_id": chunk_id}
        )
        
        # Check if any results were returned
        return len(results["documents"]) > 0
    
    def document_exists_sync(self, chunk_id: str) -> bool:
        """
        Synchronous version of document_exists.
        
        Args:
            chunk_id: The chunk_id to check
            
        Returns:
            True if the document exists, False otherwise
        """
        self._initialize_vector_store_sync()
        
        # Query for documents with matching chunk_id in metadata
        results = self._vector_store.get(
            where={"chunk_id": chunk_id}
        )
        
        # Check if any results were returned
        return len(results["documents"]) > 0 