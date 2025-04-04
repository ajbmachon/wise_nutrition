"""
Tests for embedding factory module.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock

from wise_nutrition.utils.config import Config
from wise_nutrition.embeddings.embedding_factory import (
    get_embedding_manager,
    get_document_retriever,
    get_document_retriever_sync
)
from wise_nutrition.embeddings.embedding_manager import EmbeddingManager
from wise_nutrition.embeddings.chroma_embedding_manager import ChromaEmbeddingManager


class TestEmbeddingFactory:
    """
    Test the embedding factory functions.
    """
    
    def setup_method(self):
        """
        Set up test environment.
        """
        # Mock Config
        self.mock_config = Mock(spec=Config)
        self.mock_config.vector_db_type = "weaviate"
        self.mock_config.weaviate_collection_name = "test_weaviate_collection"
        self.mock_config.chroma_collection_name = "test_chroma_collection"
        self.mock_config.chroma_persist_directory = "/tmp/test_chroma_db"
    
    @patch('wise_nutrition.embeddings.embedding_factory.EmbeddingManager')
    def test_get_embedding_manager_weaviate(self, mock_embedding_manager_class):
        """
        Test getting a Weaviate embedding manager.
        """
        # Create a mock instance
        mock_instance = Mock(spec=EmbeddingManager)
        mock_embedding_manager_class.return_value = mock_instance
        
        # Set vector_db_type to weaviate
        self.mock_config.vector_db_type = "weaviate"
        
        # Call the factory function
        manager = get_embedding_manager(config=self.mock_config)
        
        # Check that EmbeddingManager was initialized with correct parameters
        mock_embedding_manager_class.assert_called_once_with(
            config=self.mock_config,
            collection_name=self.mock_config.weaviate_collection_name
        )
        assert manager == mock_instance
    
    @patch('wise_nutrition.embeddings.embedding_factory.ChromaEmbeddingManager')
    def test_get_embedding_manager_chroma(self, mock_chroma_manager_class):
        """
        Test getting a ChromaDB embedding manager.
        """
        # Create a mock instance
        mock_instance = Mock(spec=ChromaEmbeddingManager)
        mock_chroma_manager_class.return_value = mock_instance
        
        # Set vector_db_type to chroma
        self.mock_config.vector_db_type = "chroma"
        
        # Call the factory function
        manager = get_embedding_manager(config=self.mock_config)
        
        # Check that ChromaEmbeddingManager was initialized with correct parameters
        mock_chroma_manager_class.assert_called_once_with(
            config=self.mock_config,
            collection_name=self.mock_config.chroma_collection_name,
            persist_directory=self.mock_config.chroma_persist_directory
        )
        assert manager == mock_instance
    
    @patch('wise_nutrition.embeddings.embedding_factory.ChromaEmbeddingManager')
    @patch('wise_nutrition.embeddings.embedding_factory.EmbeddingManager')
    def test_get_embedding_manager_explicit_type(self, mock_embedding_manager_class, mock_chroma_manager_class):
        """
        Test getting an embedding manager with explicitly specified type.
        """
        # Create mock instances
        mock_weaviate_instance = Mock(spec=EmbeddingManager)
        mock_embedding_manager_class.return_value = mock_weaviate_instance
        
        mock_chroma_instance = Mock(spec=ChromaEmbeddingManager)
        mock_chroma_manager_class.return_value = mock_chroma_instance
        
        # Set vector_db_type to weaviate in config
        self.mock_config.vector_db_type = "weaviate"
        
        # But explicitly request chroma
        manager = get_embedding_manager(config=self.mock_config, vector_db_type="chroma")
        
        # Check that ChromaEmbeddingManager was initialized
        mock_chroma_manager_class.assert_called_once()
        mock_embedding_manager_class.assert_not_called()
        assert manager == mock_chroma_instance
    
    @pytest.mark.asyncio
    @patch('wise_nutrition.embeddings.embedding_factory.get_embedding_manager')
    async def test_get_document_retriever_weaviate(self, mock_get_manager):
        """
        Test getting a Weaviate document retriever.
        """
        # Set up mock manager
        mock_manager = Mock(spec=EmbeddingManager)
        mock_retriever = Mock()
        mock_manager.get_retriever.return_value = mock_retriever
        mock_get_manager.return_value = mock_manager
        
        # Set vector_db_type to weaviate
        self.mock_config.vector_db_type = "weaviate"
        
        # Call the factory function
        retriever = await get_document_retriever(config=self.mock_config, k=5)
        
        # Check that get_retriever was called with correct parameters
        mock_manager.get_retriever.assert_called_once_with(k=5)
        assert retriever == mock_retriever
    
    @pytest.mark.asyncio
    @patch('wise_nutrition.embeddings.embedding_factory.get_embedding_manager')
    async def test_get_document_retriever_chroma(self, mock_get_manager):
        """
        Test getting a ChromaDB document retriever.
        """
        # Set up mock manager
        mock_manager = Mock(spec=ChromaEmbeddingManager)
        mock_retriever = Mock()
        mock_manager.get_retriever.return_value = mock_retriever
        mock_get_manager.return_value = mock_manager
        
        # Set vector_db_type to chroma
        self.mock_config.vector_db_type = "chroma"
        
        # Call the factory function
        retriever = await get_document_retriever(config=self.mock_config, k=5)
        
        # Check that get_retriever was called with correct parameters
        mock_manager.get_retriever.assert_called_once_with(k=5)
        assert retriever == mock_retriever
    
    @patch('wise_nutrition.embeddings.embedding_factory.get_embedding_manager')
    def test_get_document_retriever_sync_weaviate(self, mock_get_manager):
        """
        Test getting a Weaviate document retriever (synchronous).
        """
        # Set up mock manager
        mock_manager = Mock(spec=EmbeddingManager)
        mock_retriever = Mock()
        mock_manager.get_retriever.return_value = mock_retriever
        mock_get_manager.return_value = mock_manager
        
        # Set vector_db_type to weaviate
        self.mock_config.vector_db_type = "weaviate"
        
        # Call the factory function
        retriever = get_document_retriever_sync(config=self.mock_config, k=5)
        
        # Check that get_retriever was called with correct parameters
        mock_manager.get_retriever.assert_called_once_with(k=5)
        assert retriever == mock_retriever
    
    @patch('wise_nutrition.embeddings.embedding_factory.get_embedding_manager')
    def test_get_document_retriever_sync_chroma(self, mock_get_manager):
        """
        Test getting a ChromaDB document retriever (synchronous).
        """
        # Set up mock manager
        mock_manager = Mock(spec=ChromaEmbeddingManager)
        mock_retriever = Mock()
        mock_manager.get_retriever_sync.return_value = mock_retriever
        mock_get_manager.return_value = mock_manager
        
        # Set vector_db_type to chroma
        self.mock_config.vector_db_type = "chroma"
        
        # Call the factory function
        retriever = get_document_retriever_sync(config=self.mock_config, k=5)
        
        # Check that get_retriever_sync was called with correct parameters
        mock_manager.get_retriever_sync.assert_called_once_with(k=5)
        assert retriever == mock_retriever 