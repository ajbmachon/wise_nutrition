"""
Tests for the CLI embedding module.
"""
import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from wise_nutrition.cli.embed import embed, files
from langchain_core.documents import Document


class TestCliEmbed:
    """Test the CLI embedding functionality."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.runner = CliRunner()
        
        # Create temp files for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a text file
        self.text_file = Path(self.test_dir) / "test.txt"
        with open(self.text_file, 'w') as f:
            f.write("This is test content for embedding.")
        
        # Create a JSON file with sample data
        self.json_file = Path(self.test_dir) / "test.json"
        test_data = [
            {
                "title": "Test Recipe",
                "description": "A simple test recipe",
                "ingredients": ["ingredient1", "ingredient2"],
                "instructions": "Mix all ingredients."
            }
        ]
        with open(self.json_file, 'w') as f:
            json.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_embed_command_exists(self):
        """Test that the embed command exists."""
        result = self.runner.invoke(embed, ['--help'])
        assert result.exit_code == 0
        assert "Embed files into ChromaDB collections" in result.output
    
    def test_files_command_exists(self):
        """Test that the files subcommand exists."""
        result = self.runner.invoke(embed, ['files', '--help'])
        assert result.exit_code == 0
        assert "Embed files into ChromaDB" in result.output
    
    @patch('wise_nutrition.cli.embed._embed_files_sync')
    def test_files_command_sync_mode(self, mock_embed_sync):
        """Test the files command in sync mode."""
        # Run command
        result = self.runner.invoke(embed, [
            'files', 
            str(self.text_file), 
            '--sync-mode',
            '--collection-name', 'test_collection'
        ])
        
        # Check result
        assert result.exit_code == 0
        mock_embed_sync.assert_called_once()
        
        # Check the arguments
        args = mock_embed_sync.call_args[0]
        assert args[0] == self.text_file
        assert args[1] == 'test_collection'
    
    @patch('wise_nutrition.cli.embed.asyncio.run')
    def test_files_command_async_mode(self, mock_asyncio_run):
        """Test the files command in async mode."""
        # Run command
        result = self.runner.invoke(embed, [
            'files', 
            str(self.text_file), 
            '--async-mode',
            '--collection-name', 'test_collection'
        ])
        
        # Check result
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()
    
    def test_process_text_file(self):
        """Test processing a text file."""
        from wise_nutrition.cli.embed import _process_text_file
        
        documents = _process_text_file(self.text_file)
        
        assert len(documents) == 1
        assert documents[0].page_content == "This is test content for embedding."
        assert documents[0].metadata["chunk_id"] == f"file_{self.text_file.stem}"
    
    def test_process_json_file(self):
        """Test processing a JSON file."""
        from wise_nutrition.cli.embed import _process_json_file
        
        documents = _process_json_file(self.json_file)
        
        assert len(documents) == 1
        assert "Test Recipe" in documents[0].page_content
        assert "A simple test recipe" in documents[0].page_content
        assert documents[0].metadata["chunk_id"].startswith(f"json_{self.json_file.stem}")
        assert "title" in documents[0].metadata
        assert documents[0].metadata["title"] == "Test Recipe"
        
    @patch('wise_nutrition.cli.embed.ChromaEmbeddingManager')
    def test_embed_files_sync_functionality(self, mock_chroma_manager):
        """Test the _embed_files_sync function with proper mocking."""
        from wise_nutrition.cli.embed import _embed_files_sync
        
        # Create mock documents
        test_docs = [
            Document(page_content="Test doc", metadata={"chunk_id": "test1"})
        ]
        
        # Mock ChromaEmbeddingManager instance
        mock_manager = MagicMock()
        mock_chroma_manager.return_value = mock_manager
        
        # Mock load_documents_sync to return our test documents
        with patch('wise_nutrition.cli.embed.load_documents_sync', return_value=test_docs):
            # Call the function with our test directory
            _embed_files_sync(
                file_path=self.test_dir,
                collection_name="test_collection", 
                persist_dir="/tmp/chroma",
                clear_existing=True
            )
        
        # Check that the manager was instantiated with correct parameters
        mock_chroma_manager.assert_called_once()
        # Check that create_collection_sync was called because clear_existing=True
        mock_manager.create_collection_sync.assert_called_once()
        # Check that add_documents_sync was called with our test documents
        mock_manager.add_documents_sync.assert_called_once_with(test_docs) 