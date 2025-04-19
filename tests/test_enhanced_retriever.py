"""
Integration tests for the enhanced retriever with query reformulation.
"""
import unittest
from unittest.mock import MagicMock, patch
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.language_models import BaseLLM

from wise_nutrition.query_reformulation import QueryReformulator
from wise_nutrition.enhanced_retriever import EnhancedNutritionRetriever

class MockBaseRetriever(BaseRetriever):
    """Mock base retriever for testing."""
    
    def _get_relevant_documents(self, query, *, run_manager):
        """Mock implementation that returns predefined documents based on the query."""
        docs = []
        
        if "vitamin c" in query.lower():
            docs.append(Document(
                page_content="Vitamin C is important for immune function.",
                metadata={"source": "test", "type": "vitamin"}
            ))
        
        if "citrus" in query.lower():
            docs.append(Document(
                page_content="Citrus fruits are rich in vitamin C.",
                metadata={"source": "test", "type": "food"}
            ))
            
        if "immune" in query.lower():
            docs.append(Document(
                page_content="The immune system protects the body from infections.",
                metadata={"source": "test", "type": "general"}
            ))
            
        if "health" in query.lower():
            docs.append(Document(
                page_content="Vitamins are essential for overall health.",
                metadata={"source": "test", "type": "general"}
            ))
        
        return docs


class TestEnhancedNutritionRetriever(unittest.TestCase):
    """Test the EnhancedNutritionRetriever class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a mock LLM
        self.mock_llm = MagicMock(spec=BaseLLM)
        self.mock_llm.predict.return_value = """What foods contain vitamin C?
What role does vitamin C play in immune function?
What are the health benefits of consuming citrus fruits?
How does vitamin C support overall health and wellness?"""
        
        # Create a mock base retriever
        self.base_retriever = MockBaseRetriever()
        
        # Create a mock callback manager
        self.mock_callback_manager = MagicMock(spec=CallbackManagerForRetrieverRun)
        self.mock_callback_manager.get_child.return_value = None
        
        # Create a query reformulator
        self.query_reformulator = QueryReformulator(llm=self.mock_llm)
        
        # Create the enhanced retriever
        self.enhanced_retriever = EnhancedNutritionRetriever(
            base_retriever=self.base_retriever,
            query_reformulator=self.query_reformulator,
            k=4,
            max_queries=4
        )
    
    def test_get_relevant_documents(self):
        """Test retrieving documents with query reformulation."""
        # Test with a simple query
        query = "Tell me about vitamin C"
        docs = self.enhanced_retriever._get_relevant_documents(
            query, run_manager=self.mock_callback_manager
        )
        
        # Check that we got documents from multiple perspectives
        self.assertGreater(len(docs), 0)
        
        # We should get documents about vitamin C, immune function, citrus, and health
        page_contents = [doc.page_content for doc in docs]
        
        # Check if we have more diverse results than just using the original query
        # Due to reformulation, we should find documents mentioning citrus, immune function, etc.
        # even though the original query didn't mention these terms
        self.assertTrue(
            any("citrus" in content.lower() for content in page_contents) or
            any("immune" in content.lower() for content in page_contents) or
            any("health" in content.lower() for content in page_contents)
        )
    
    def test_reformulation_disabled(self):
        """Test retrieval with reformulation disabled."""
        # Disable reformulation
        self.enhanced_retriever.use_reformulation = False
        
        # Test with a simple query
        query = "Tell me about vitamin C"
        docs = self.enhanced_retriever._get_relevant_documents(
            query, run_manager=self.mock_callback_manager
        )
        
        # Check that we get only documents from the original query
        page_contents = [doc.page_content for doc in docs]
        
        # Should only have results about vitamin C, not citrus (since reformulation is disabled)
        self.assertTrue(any("vitamin c" in content.lower() for content in page_contents))
        self.assertFalse(any("citrus" in content.lower() for content in page_contents))
    
    def test_deduplicate_documents(self):
        """Test document deduplication."""
        # Create duplicate documents
        docs = [
            Document(page_content="Document 1", metadata={"id": 1}),
            Document(page_content="Document 2", metadata={"id": 2}),
            Document(page_content="Document 1", metadata={"id": 3}),  # Duplicate content
            Document(page_content="Document 3", metadata={"id": 4})
        ]
        
        # Deduplicate
        unique_docs = self.enhanced_retriever._deduplicate_documents(docs)
        
        # Check that we have the correct number of unique documents
        self.assertEqual(len(unique_docs), 3)
        
        # Check that the deduplicated documents have the expected content
        contents = [doc.page_content for doc in unique_docs]
        self.assertIn("Document 1", contents)
        self.assertIn("Document 2", contents)
        self.assertIn("Document 3", contents)
    
    def test_from_llm_factory_method(self):
        """Test creating an enhanced retriever using the from_llm factory method."""
        # Create a retriever using the factory method
        retriever = EnhancedNutritionRetriever.from_llm(
            base_retriever=self.base_retriever,
            llm=self.mock_llm,
            k=3,
            max_queries=2
        )
        
        # Check that the retriever was created correctly
        self.assertIsInstance(retriever, EnhancedNutritionRetriever)
        self.assertEqual(retriever.k, 3)
        self.assertEqual(retriever.max_queries, 2)
        self.assertIsNotNone(retriever.query_reformulator)
        self.assertTrue(retriever.use_reformulation)


if __name__ == "__main__":
    unittest.main() 