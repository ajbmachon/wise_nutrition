"""
Tests for the ChromaDB embedding manager.
"""
import os
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock, Mock

from langchain_core.documents import Document
from wise_nutrition.embeddings.chroma_embedding_manager import ChromaEmbeddingManager
from wise_nutrition.utils.config import Config


class TestChromaEmbeddingManager:
    """
    Test the ChromaEmbeddingManager class.
    """
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock Config
        self.mock_config = Mock(spec=Config)
        self.mock_config.openai_api_key = "sk-test-key"
        self.mock_config.chroma_collection_name = "test_collection"
        
        # Create ChromaEmbeddingManager with mocked dependencies
        with patch('langchain_openai.OpenAIEmbeddings'):
            with patch('langchain_chroma.Chroma'):
                self.embedding_manager = ChromaEmbeddingManager(
                    config=self.mock_config,
                    collection_name="test_collection",
                    persist_directory=self.temp_dir
                )
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization with custom parameters."""
        with patch('langchain_openai.OpenAIEmbeddings'):
            with patch('langchain_chroma.Chroma'):
                manager = ChromaEmbeddingManager(
                    config=self.mock_config,
                    collection_name="custom_collection",
                    persist_directory=self.temp_dir
                )
                
                assert manager._collection_name == "custom_collection"
                assert manager._persist_directory == self.temp_dir
    
    @pytest.mark.asyncio
    async def test_create_collection(self):
        """Test creating a ChromaDB collection."""
        with patch('os.path.exists', return_value=True):
            with patch('shutil.rmtree') as mock_rmtree:
                with patch('os.makedirs') as mock_makedirs:
                    with patch('langchain_chroma.Chroma'):
                        await self.embedding_manager.create_collection()
                        
                        # Check that existing directory was removed
                        mock_rmtree.assert_called_once_with(self.temp_dir)
                        
                        # Check that directory was created
                        mock_makedirs.assert_called_once_with(self.temp_dir, exist_ok=True)
    
    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding documents to the ChromaDB collection."""
        # Create test documents
        documents = [
            Document(page_content="Test document 1", metadata={"chunk_id": "doc1"}),
            Document(page_content="Test document 2", metadata={"chunk_id": "doc2"})
        ]
        
        # Mock vector store
        mock_vector_store = MagicMock()
        self.embedding_manager._vector_store = mock_vector_store
        
        # Test adding documents
        await self.embedding_manager.add_documents(documents)
        
        # Check that documents were added to the vector store
        mock_vector_store.add_documents.assert_called_once_with(documents)
        # No need to check persist() as it's not called anymore
    
    @pytest.mark.asyncio
    async def test_initialize_vector_store_existing(self):
        """Test initializing vector store when the collection already exists."""
        with patch('os.path.exists', return_value=True):
            with patch('wise_nutrition.embeddings.chroma_embedding_manager.Chroma') as mock_chroma:
                # Mock for the Chroma constructor
                mock_chroma_instance = MagicMock()
                mock_chroma.return_value = mock_chroma_instance
                
                # Reset vector store to None to test initialization
                self.embedding_manager._vector_store = None
                
                # Test initialization
                result = await self.embedding_manager._initialize_vector_store()
                
                # Check that Chroma was initialized correctly
                mock_chroma.assert_called_once()
                assert result == mock_chroma_instance
    
    @pytest.mark.asyncio
    async def test_get_retriever(self):
        """Test getting a retriever from the embedding manager."""
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.embedding_manager._vector_store = mock_vector_store
        
        # Test getting retriever
        result = await self.embedding_manager.get_retriever(k=5)
        
        # Check that retriever was created with correct parameters
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert result == mock_retriever
    
    @pytest.mark.asyncio
    async def test_search_similar(self):
        """Test searching for similar documents."""
        # Mock retriever
        mock_retriever = MagicMock()
        mock_docs = [Document(page_content="Result 1"), Document(page_content="Result 2")]
        mock_retriever.get_relevant_documents.return_value = mock_docs
        
        # Mock get_retriever
        with patch.object(self.embedding_manager, 'get_retriever', return_value=mock_retriever):
            # Test search
            results = await self.embedding_manager.search_similar("test query", k=3)
            
            # Check that retriever was used correctly
            mock_retriever.get_relevant_documents.assert_called_once_with("test query")
            assert results == mock_docs
    
    @pytest.mark.asyncio
    async def test_document_exists(self):
        """Test checking if a document exists."""
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.get.return_value = {"documents": ["doc1"]}
        self.embedding_manager._vector_store = mock_vector_store
        
        # Test document exists
        result = await self.embedding_manager.document_exists("test_id")
        
        # Check that get was called with correct parameters
        mock_vector_store.get.assert_called_once_with(where={"chunk_id": "test_id"})
        assert result is True
        
        # Test document does not exist
        mock_vector_store.get.return_value = {"documents": []}
        result = await self.embedding_manager.document_exists("nonexistent_id")
        assert result is False 