"""
Tests for the RAG chain.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from wise_nutrition.rag_chain import NutritionRAGChain
from wise_nutrition.memory import ConversationMemoryManager
from langgraph.checkpoint.memory import MemorySaver


class TestNutritionRAGChain:
    """
    Test the NutritionRAGChain class.
    """
    
    def setup_method(self):
        """Set up the test environment."""
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_memory_saver = MagicMock(spec=MemorySaver)
        self.mock_memory_manager = MagicMock(spec=ConversationMemoryManager)
        # Configure the memory manager to return our mock memory saver
        self.mock_memory_manager.get_memory_saver.return_value = self.mock_memory_saver
        
        self.rag_chain = NutritionRAGChain(
            retriever=self.mock_retriever,
            llm=self.mock_llm,
            memory_manager=self.mock_memory_manager
        )
    
    def test_init(self):
        """Test initialization with custom parameters."""
        # Create a memory manager for testing
        memory_manager = ConversationMemoryManager()
        
        chain = NutritionRAGChain(
            retriever=self.mock_retriever,
            llm=self.mock_llm,
            memory_manager=memory_manager,
            model_name="gpt-4"
        )
        assert chain.model_name == "gpt-4"
        assert chain.retriever == self.mock_retriever
        assert chain.llm == self.mock_llm
        assert chain.memory_manager == memory_manager
    
    def test_format_docs(self):
        """Test formatting documents."""
        # Create test documents
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Test content 1"),
            Document(page_content="Test content 2")
        ]
        
        formatted = self.rag_chain._format_docs(docs)
        expected = "Test content 1\n\nTest content 2"
        assert formatted == expected
        
        # Test empty docs
        # Test _format_docs here
    
    def test_get_memory_key(self):
        """Test generating a memory key."""
        # Test get_memory_key here
    
    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    @patch('langchain_core.prompts.ChatPromptTemplate')
    @patch('langchain_core.runnables.RunnableWithMessageHistory')
    async def test_build_chain(self, mock_runnable_with_history, mock_prompt, mock_chat):
        """Test building the RAG chain with LangGraph memory."""
        # Test build_chain here
    
    @pytest.mark.asyncio
    async def test_invoke(self):
        """Test invoking the RAG chain."""
        # Setup
        session_id = "test-session-id"
        self.rag_chain.build_chain = MagicMock()
        self.rag_chain.get_memory_key = MagicMock(return_value=f"test-memory-key-{session_id}")
        mock_chain = MagicMock()
        self.rag_chain.build_chain.return_value = mock_chain
        mock_chain.invoke.return_value = {"answer": "Test answer", "sources": ["Test source"]}
        
        # Test invoke with session_id here 