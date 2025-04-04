"""
Factory for creating embedding managers.
"""
from typing import Optional, Union, Callable, Any

from wise_nutrition.utils.config import Config
from wise_nutrition.embeddings.embedding_manager import EmbeddingManager
from wise_nutrition.embeddings.chroma_embedding_manager import ChromaEmbeddingManager


def get_embedding_manager(
    config: Optional[Config] = None,
    vector_db_type: Optional[str] = None,
    async_mode: bool = False
) -> Union[EmbeddingManager, ChromaEmbeddingManager]:
    """
    Factory method to get the appropriate embedding manager based on configuration.
    
    Args:
        config: Configuration object
        vector_db_type: Explicitly specify vector database type ('weaviate' or 'chroma')
        async_mode: Whether to use async methods (only affects ChromaDB)
        
    Returns:
        An embedding manager instance
    """
    config = config or Config()
    db_type = vector_db_type or config.vector_db_type
    
    if db_type.lower() == 'chroma':
        return ChromaEmbeddingManager(
            config=config,
            collection_name=config.chroma_collection_name,
            persist_directory=config.chroma_persist_directory
        )
    else:
        # Default to Weaviate
        return EmbeddingManager(
            config=config,
            collection_name=config.weaviate_collection_name
        )


async def get_document_retriever(
    config: Optional[Config] = None,
    vector_db_type: Optional[str] = None,
    k: int = 4
) -> Any:
    """
    Get a document retriever based on the configured vector store.
    
    Args:
        config: Configuration object
        vector_db_type: Explicitly specify vector database type ('weaviate' or 'chroma')
        k: Number of documents to retrieve
        
    Returns:
        A document retriever
    """
    config = config or Config()
    db_type = vector_db_type or config.vector_db_type
    
    manager = get_embedding_manager(config, db_type, async_mode=True)
    
    if db_type.lower() == 'chroma':
        return await manager.get_retriever(k=k)
    else:
        return manager.get_retriever(k=k)


def get_document_retriever_sync(
    config: Optional[Config] = None,
    vector_db_type: Optional[str] = None,
    k: int = 4
) -> Any:
    """
    Get a document retriever based on the configured vector store (synchronous version).
    
    Args:
        config: Configuration object
        vector_db_type: Explicitly specify vector database type ('weaviate' or 'chroma')
        k: Number of documents to retrieve
        
    Returns:
        A document retriever
    """
    config = config or Config()
    db_type = vector_db_type or config.vector_db_type
    
    manager = get_embedding_manager(config, db_type, async_mode=False)
    
    if db_type.lower() == 'chroma':
        return manager.get_retriever_sync(k=k)
    else:
        return manager.get_retriever(k=k) 