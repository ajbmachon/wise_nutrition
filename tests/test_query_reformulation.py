"""
Unit tests for the query reformulation module.
"""
import unittest
from unittest.mock import MagicMock, patch
from typing import List

from langchain_core.language_models import BaseLLM
from wise_nutrition.query_reformulation import QueryReformulator, LineListOutputParser

class TestLineListOutputParser(unittest.TestCase):
    """Test the LineListOutputParser class."""
    
    def setUp(self):
        self.parser = LineListOutputParser()
    
    def test_parse_clean_lines(self):
        """Test parsing clean lines with no numbering."""
        text = "Line 1\nLine 2\nLine 3"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Line 1", "Line 2", "Line 3"])
    
    def test_parse_with_empty_lines(self):
        """Test parsing text with empty lines."""
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Line 1", "Line 2", "Line 3"])
    
    def test_parse_with_numbering(self):
        """Test parsing text with numerical prefixes."""
        text = "1. Line 1\n2. Line 2\n3. Line 3"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Line 1", "Line 2", "Line 3"])
        
        text = "1) Line 1\n2) Line 2\n3) Line 3"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Line 1", "Line 2", "Line 3"])

class TestQueryReformulator(unittest.TestCase):
    """Test the QueryReformulator class."""
    
    def setUp(self):
        # Create a mock LLM for testing
        self.mock_llm = MagicMock(spec=BaseLLM)
        self.mock_llm.predict.return_value = "Query 1\nQuery 2\nQuery 3\nQuery 4"
        
        # Create a query reformulator with the mock LLM
        self.reformulator = QueryReformulator(llm=self.mock_llm)
    
    def test_rewrite_query(self):
        """Test rewriting a query."""
        # Test with include_original=True
        self.reformulator.include_original = True
        result = self.reformulator.rewrite_query("What are the benefits of vitamin C?")
        
        # Check that the original query is included and the LLM was called correctly
        self.assertEqual(len(result), 5)  # Original + 4 generated
        self.assertEqual(result[0], "What are the benefits of vitamin C?")
        self.assertIn("Query 1", result)
        self.assertIn("Query 2", result)
        self.assertIn("Query 3", result)
        self.assertIn("Query 4", result)
        
        # Test with include_original=False
        self.reformulator.include_original = False
        result = self.reformulator.rewrite_query("What are the benefits of vitamin C?")
        
        # Check that only generated queries are included
        self.assertEqual(len(result), 4)  # 4 generated, no original
        self.assertIn("Query 1", result)
        self.assertIn("Query 2", result)
        self.assertIn("Query 3", result)
        self.assertIn("Query 4", result)
    
    def test_as_runnable(self):
        """Test converting to a runnable lambda."""
        runnable = self.reformulator.as_runnable()
        self.assertIsNotNone(runnable)
        
        # Test invoking the runnable
        result = runnable.invoke("What are the benefits of vitamin C?")
        self.assertEqual(len(result), 5)  # Original + 4 generated

if __name__ == "__main__":
    unittest.main() 