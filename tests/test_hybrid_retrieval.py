"""
Tests for the hybrid retrieval strategy implementation.

This tests the NutritionRetriever's hybrid retrieval capabilities, ensuring
that it properly combines semantic search, keyword matching, and metadata-based
filtering to provide the most relevant results.
"""
import unittest
from unittest.mock import MagicMock, patch
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from wise_nutrition.retriever import NutritionRetriever
from wise_nutrition.reranker import DocumentReRanker, ReRankingConfig


class MockBaseRetriever(BaseRetriever):
    """Mock base retriever for testing."""
    
    def _get_relevant_documents(self, query, *, run_manager):
        """Mock implementation that returns predefined documents based on the query."""
        docs = [
            Document(
                page_content="Vitamin D is essential for calcium absorption and bone health.",
                metadata={"source": "nih.gov", "type": "vitamin", "name": "Vitamin D"}
            ),
            Document(
                page_content="High protein foods include chicken, fish, and legumes.",
                metadata={"source": "nutrition.org", "type": "food", "name": "Protein Sources"}
            ),
            Document(
                page_content="Iron deficiency is common and can lead to anemia.",
                metadata={"source": "mayoclinic.org", "type": "mineral", "name": "Iron"}
            ),
            Document(
                page_content="A balanced diet contains proteins, carbohydrates, and fats.",
                metadata={"source": "health.org", "type": "general", "name": "Balanced Diet"}
            ),
            Document(
                page_content="Vitamin C supports immune function and is found in citrus fruits.",
                metadata={"source": "cdc.gov", "type": "vitamin", "name": "Vitamin C"}
            ),
        ]
        return docs


class TestHybridRetrieval(unittest.TestCase):
    """Test the hybrid retrieval capabilities of NutritionRetriever."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a mock base retriever
        self.base_retriever = MockBaseRetriever()
        
        # Create a mock callback manager
        self.mock_callback_manager = MagicMock(spec=CallbackManagerForRetrieverRun)
        self.mock_callback_manager.get_child.return_value = None
        
        # Create a nutrition retriever
        self.nutrition_retriever = NutritionRetriever(
            base_retriever=self.base_retriever,
            k=3
        )
    
    def test_query_intent_detection(self):
        """Test the query intent detection functionality."""
        # Test nutrient info intent
        nutrient_query = "What are the benefits of vitamin D?"
        intent = self.nutrition_retriever._detect_query_intent(nutrient_query)
        self.assertGreater(intent["nutrient_info"], 0)
        
        # Test food sources intent
        food_query = "What foods are high in protein?"
        intent = self.nutrition_retriever._detect_query_intent(food_query)
        self.assertGreater(intent["food_sources"], 0)
        
        # Test health condition intent
        health_query = "How to prevent iron deficiency?"
        intent = self.nutrition_retriever._detect_query_intent(health_query)
        self.assertGreater(intent["health_condition"], 0)
    
    def test_keyword_scoring(self):
        """Test the keyword-based scoring of documents."""
        docs = self.base_retriever.get_relevant_documents("", run_manager=self.mock_callback_manager)
        
        # Test for a query about vitamin D
        query = "vitamin D benefits"
        scores = self.nutrition_retriever._score_by_keywords(docs, query)
        
        # The first document (about vitamin D) should have the highest score
        highest_score_index = scores.index(max(scores))
        self.assertEqual(highest_score_index, 0)
        
        # Test for a query about protein
        query = "protein sources"
        scores = self.nutrition_retriever._score_by_keywords(docs, query)
        
        # The second document (about protein) should have the highest score
        highest_score_index = scores.index(max(scores))
        self.assertEqual(highest_score_index, 1)
    
    def test_metadata_boost(self):
        """Test the metadata-based boosting of documents."""
        docs = self.base_retriever.get_relevant_documents("", run_manager=self.mock_callback_manager)
        
        # Simulate a nutrient info intent
        intent = {"nutrient_info": 0.8, "food_sources": 0.0, "health_condition": 0.0}
        
        # Get metadata boost for the vitamin D document
        boost = self.nutrition_retriever._get_metadata_boost(docs[0], intent)
        self.assertGreater(boost, 0)
        
        # Simulate a health condition intent
        intent = {"nutrient_info": 0.0, "food_sources": 0.0, "health_condition": 0.8}
        
        # Health condition queries should boost authoritative sources
        boost_nih = self.nutrition_retriever._get_metadata_boost(docs[0], intent)  # nih.gov
        boost_mayo = self.nutrition_retriever._get_metadata_boost(docs[2], intent)  # mayoclinic.org
        boost_health = self.nutrition_retriever._get_metadata_boost(docs[3], intent)  # health.org
        
        # NIH and Mayo Clinic should get higher boosts than generic health.org
        self.assertGreater(boost_nih, boost_health)
        self.assertGreater(boost_mayo, boost_health)
    
    def test_hybrid_retrieval_end_to_end(self):
        """Test the complete hybrid retrieval pipeline."""
        # Test retrieval for a vitamin D query
        query = "What are the health benefits of vitamin D?"
        docs = self.nutrition_retriever._get_relevant_documents(
            query, run_manager=self.mock_callback_manager
        )
        
        # The vitamin D document should be the first result
        self.assertIn("vitamin d", docs[0].page_content.lower())
        
        # Test retrieval for a protein query
        query = "Good sources of protein in diet"
        docs = self.nutrition_retriever._get_relevant_documents(
            query, run_manager=self.mock_callback_manager
        )
        
        # The protein document should be the first result
        self.assertIn("protein", docs[0].page_content.lower())
    
    def test_with_reranker(self):
        """Test retrieval with the reranker enabled."""
        # Create a retriever with reranking
        reranking_config = ReRankingConfig(
            semantic_weight=0.5,
            authority_weight=0.3,
            term_proximity_weight=0.2
        )
        
        retriever_with_reranker = NutritionRetriever.with_reranker(
            base_retriever=self.base_retriever,
            k=3,
            reranking_config=reranking_config
        )
        
        # Test retrieval with reranking
        query = "vitamin deficiency health impacts"
        docs = retriever_with_reranker._get_relevant_documents(
            query, run_manager=self.mock_callback_manager
        )
        
        # We should get at least one result containing "vitamin"
        self.assertTrue(any("vitamin" in doc.page_content.lower() for doc in docs))
        
        # The highest authority sources should be preferred
        # Check if NIH or CDC documents appear first
        first_doc_source = docs[0].metadata.get("source", "").lower()
        self.assertTrue(any(src in first_doc_source for src in ["nih.gov", "cdc.gov", "mayoclinic.org"]))


if __name__ == "__main__":
    unittest.main() 