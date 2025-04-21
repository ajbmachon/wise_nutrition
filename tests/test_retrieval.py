"""Tests for the retrieval module."""

import json
import os
from pathlib import Path
import pytest
from typing import Dict, Any, List
import tempfile

from wise_nutrition.retrieval import SimpleFileVectorStore, RetrievalService
from wise_nutrition.models import NutrientInfo, Recipe, NutritionTheory


@pytest.fixture
def sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample data for testing.
    
    Returns:
        Dictionary containing sample data
    """
    return {
        "vitamins": [
            {
                "text": "Vitamin D is produced in the skin when exposed to sunlight.",
                "metadata": {
                    "nutrient": "Vitamin D",
                    "category": "sources"
                }
            },
            {
                "text": "Vitamin D deficiency may cause bone pain and muscle weakness.",
                "metadata": {
                    "nutrient": "Vitamin D",
                    "category": "symptoms"
                }
            }
        ],
        "theory": [
            {
                "section": "FAT-SOLUBLE VITAMINS",
                "quote": "Vitamin D is found only in animal fats.",
                "summary": "Information about vitamin D sources.",
                "tags": ["vitamin D", "animal fats"],
                "source": "Nourishing Traditions, p. 42",
                "chunk_id": "theory_fats_42_01"
            }
        ],
        "recipes": [
            {
                "name": "Salmon with Lemon",
                "description": "Simple salmon dish rich in vitamin D and omega-3 fatty acids.",
                "ingredients": ["salmon", "lemon", "salt", "olive oil"],
                "instructions": "Bake salmon with lemon, salt, and olive oil.",
                "nutrition_benefits": ["Rich in vitamin D", "High in omega-3 fatty acids"],
                "tags": ["high-vitamin-D", "seafood", "omega-3"]
            }
        ]
    }


@pytest.fixture
def temp_data_dir(sample_data: Dict[str, List[Dict[str, Any]]]) -> Path:
    """Create a temporary data directory with sample data.
    
    Args:
        sample_data: Sample data to write to files
        
    Returns:
        Path to temporary data directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        samples_dir = data_dir / "samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        # Write sample data files
        for file_name, data in sample_data.items():
            with open(samples_dir / f"{file_name}.json", "w") as f:
                json.dump(data, f)
        
        yield data_dir


@pytest.mark.asyncio
async def test_simple_file_vector_store_search(temp_data_dir: Path):
    """Test SimpleFileVectorStore search with expected query.
    
    Args:
        temp_data_dir: Temporary data directory with sample data
    """
    # Initialize store with test data
    store = SimpleFileVectorStore(temp_data_dir)
    
    # Test search with simple query
    results = await store.search("vitamin D", limit=5)
    
    # Verify we get results
    assert len(results) > 0
    
    # Verify we got results from different categories
    result_types = {result["type"] for result in results}
    assert "nutrient_info" in result_types
    
    # Verify the highest scoring result is relevant
    assert "vitamin d" in results[0]["item"]["text"].lower()


@pytest.mark.asyncio
async def test_simple_file_vector_store_empty_query(temp_data_dir: Path):
    """Test SimpleFileVectorStore with empty query (edge case).
    
    Args:
        temp_data_dir: Temporary data directory with sample data
    """
    # Initialize store with test data
    store = SimpleFileVectorStore(temp_data_dir)
    
    # Test search with empty query
    results = await store.search("", limit=5)
    
    # Verify we get empty results for empty query
    assert len(results) == 0


@pytest.mark.asyncio
async def test_retrieval_service(temp_data_dir: Path):
    """Test RetrievalService with expected query.
    
    Args:
        temp_data_dir: Temporary data directory with sample data
    """
    # Initialize store and service
    store = SimpleFileVectorStore(temp_data_dir)
    service = RetrievalService(store)
    
    # Test retrieve context
    context = await service.retrieve_context("vitamin D", limit=5)
    
    # Verify we have nutrition info
    assert len(context.nutrition_info) > 0
    assert all(isinstance(item, NutrientInfo) for item in context.nutrition_info)
    
    # Check if the model conversion was correct
    assert context.nutrition_info[0].metadata.nutrient == "Vitamin D"


@pytest.mark.asyncio
async def test_retrieval_service_no_results(temp_data_dir: Path):
    """Test RetrievalService with query that produces no results (failure case).
    
    Args:
        temp_data_dir: Temporary data directory with sample data
    """
    # Initialize store and service
    store = SimpleFileVectorStore(temp_data_dir)
    service = RetrievalService(store)
    
    # Test retrieve context with query that won't match anything
    context = await service.retrieve_context("xyz123", limit=5)
    
    # Verify we have empty results
    assert len(context.nutrition_info) == 0
    assert len(context.recipes) == 0
    assert len(context.theory) == 0
