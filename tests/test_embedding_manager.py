"""
Tests for the embedding manager.
"""
import os
import pytest
from unittest.mock import patch, MagicMock, Mock

from wise_nutrition.embeddings.embedding_manager import EmbeddingManager
from wise_nutrition.utils.config import Config


class TestEmbeddingManager:
    """
    Test the EmbeddingManager class.
    """
    
    def setup_method(self):
        """Set up the test environment."""
        self.mock_weaviate_url = "http://localhost:8080"
        self.mock_openai_api_key = "sk-test-key"
        self.mock_weaviate_api_key = "weaviate-test-key"
        
        # Create a mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.weaviate_url = self.mock_weaviate_url
        self.mock_config.openai_api_key = self.mock_openai_api_key
        self.mock_config.weaviate_api_key = self.mock_weaviate_api_key
        self.mock_config.weaviate_collection_name = "test_collection"
        
        with patch('weaviate.Client'):
            self.embedding_manager = EmbeddingManager(
                config=self.mock_config
            )
    
    def test_init(self):
        """Test initialization with custom parameters."""
        with patch('weaviate.Client'):
            manager = EmbeddingManager(
                config=self.mock_config,
                collection_name="CustomCollection"
            )
            assert manager._collection_name == "CustomCollection"
    
    @pytest.mark.asyncio
    @patch('weaviate.Client')
    async def test_create_collection(self, mock_weaviate_client):
        """Test creating a Weaviate collection."""
        # Set up mocks for the Weaviate client
        mock_schema = MagicMock()
        # Replace the _client instance with our mock
        self.embedding_manager._client = mock_weaviate_client.return_value
        # Set the schema attribute on the mock client
        self.embedding_manager._client.schema = mock_schema
        mock_schema.exists.return_value = True
        
        # Call create_collection
        self.embedding_manager.create_collection()
        
        # Verify collection was deleted and created
        mock_schema.exists.assert_called_once_with(self.embedding_manager._collection_name)
        mock_schema.delete_class.assert_called_once_with(self.embedding_manager._collection_name)
        mock_schema.create.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('weaviate.Client')
    @patch('langchain_openai.OpenAIEmbeddings')
    async def test_add_documents(self, mock_embeddings, mock_weaviate_client):
        """Test adding documents to the Weaviate collection."""
        # Mock vector store
        mock_vector_store = MagicMock()
        self.embedding_manager._vector_store = mock_vector_store
        
        # Mock documents
        mock_docs = [MagicMock(), MagicMock()]
        
        # Call add_documents
        self.embedding_manager.add_documents(mock_docs)
        
        # Verify documents were added
        mock_vector_store.add_documents.assert_called_once_with(mock_docs)
    
    @pytest.mark.asyncio
    @patch('weaviate.Client')
    @patch('langchain_community.vectorstores.Weaviate')
    async def test_get_retriever(self, mock_weaviate_store, mock_weaviate_client):
        """Test getting a retriever from the embedding manager."""
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.embedding_manager._vector_store = mock_vector_store
        
        # Call get_retriever
        result = self.embedding_manager.get_retriever(k=5)
        
        # Verify retriever was created with correct parameters
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert result == mock_retriever 