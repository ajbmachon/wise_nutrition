"""
Tests for the citation generation system.
"""
import unittest
from datetime import datetime

from langchain_core.documents import Document
from wise_nutrition.citation_generator import CitationGenerator, Citation


class TestCitationGenerator(unittest.TestCase):
    """Test the citation generation functionality."""
    
    def setUp(self):
        """Set up test cases."""
        self.citation_generator = CitationGenerator()
        
        # Sample documents
        self.sample_doc1 = Document(
            page_content="Vitamin D is essential for calcium absorption and bone health.",
            metadata={
                "source": "National Institutes of Health",
                "url": "https://nih.gov/vitamind",
                "type": "vitamin"
            }
        )
        
        self.sample_doc2 = Document(
            page_content="High protein foods include chicken, fish, and legumes.",
            metadata={
                "source": "American Dietetic Association",
                "name": "Protein Guide",
                "type": "food"
            }
        )
        
        self.sample_doc3 = Document(
            page_content="Iron deficiency is common and can lead to anemia.",
            metadata={}  # Minimal metadata
        )
    
    def test_generate_single_citation(self):
        """Test generating a citation for a single document."""
        citation = self.citation_generator.generate_citation(self.sample_doc1)
        
        # Verify citation fields
        self.assertEqual(citation.source_name, "National Institutes of Health")
        self.assertEqual(citation.source_url, "https://nih.gov/vitamind")
        self.assertIsNotNone(citation.date_accessed)
        self.assertIsNotNone(citation.text)
        self.assertEqual(citation.original_content, "Vitamin D is essential for calcium absorption and bone health.")
    
    def test_generate_multiple_citations(self):
        """Test generating citations for multiple documents."""
        docs = [self.sample_doc1, self.sample_doc2, self.sample_doc3]
        citations = self.citation_generator.generate_citations(docs)
        
        # Verify we have the right number of citations
        self.assertEqual(len(citations), 3)
        
        # Verify each citation matches the corresponding document
        self.assertEqual(citations[0].source_name, "National Institutes of Health")
        self.assertEqual(citations[1].source_name, "American Dietetic Association")
        self.assertEqual(citations[2].source_name, "Unknown Source")
    
    def test_citation_mla_format(self):
        """Test MLA formatting of citations."""
        citation = self.citation_generator.generate_citation(self.sample_doc1)
        mla_format = citation.to_display_format(style="mla")
        
        # Verify MLA format has the right structure
        self.assertIn("National Institutes of Health", mla_format)
        self.assertIn("https://nih.gov/vitamind", mla_format)
        self.assertIn("Accessed", mla_format)
    
    def test_citation_apa_format(self):
        """Test APA formatting of citations."""
        citation = self.citation_generator.generate_citation(self.sample_doc1)
        apa_format = citation.to_display_format(style="apa")
        
        # Verify APA format has the right structure
        self.assertIn("National Institutes of Health", apa_format)
        self.assertIn("Retrieved from https://nih.gov/vitamind", apa_format)
        self.assertIn("on", apa_format)
    
    def test_citation_chicago_format(self):
        """Test Chicago formatting of citations."""
        citation = self.citation_generator.generate_citation(self.sample_doc1)
        chicago_format = citation.to_display_format(style="chicago")
        
        # Verify Chicago format has the right structure
        self.assertIn("National Institutes of Health", chicago_format)
        self.assertIn("https://nih.gov/vitamind", chicago_format)
        self.assertIn("accessed", chicago_format)
    
    def test_citation_with_minimal_metadata(self):
        """Test generating a citation with minimal metadata."""
        citation = self.citation_generator.generate_citation(self.sample_doc3)
        
        # Verify default values are used
        self.assertEqual(citation.source_name, "Unknown Source")
        self.assertIsNone(citation.source_url)
        self.assertIsNotNone(citation.date_accessed)
    
    def test_as_runnable(self):
        """Test the as_runnable method."""
        runnable = self.citation_generator.as_runnable()
        
        # Test invoking the runnable
        result = runnable.invoke({"documents": [self.sample_doc1, self.sample_doc2]})
        
        # Verify result is a list of Citation objects
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Citation)
        self.assertIsInstance(result[1], Citation)
        
        # Verify the citations have the correct source names
        self.assertEqual(result[0].source_name, "National Institutes of Health")
        self.assertEqual(result[1].source_name, "American Dietetic Association")


if __name__ == "__main__":
    unittest.main() 