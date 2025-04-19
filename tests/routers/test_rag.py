"""
Tests for the RAG router endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from wise_nutrition.api import app
from wise_nutrition.rag_chain import RAGInput, RAGOutput

client = TestClient(app)

def test_rag_info_endpoint():
    """Test the basic info endpoint for the RAG chain."""
    response = client.get("/api/v1/nutrition_rag_chain")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "Nutrition RAG Chain"

def test_rag_playground_endpoint():
    """Test the playground endpoint for the RAG chain."""
    response = client.get("/api/v1/nutrition_rag_chain/playground")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

@pytest.mark.parametrize(
    "test_input",
    [
        {"query": "What is vitamin D?", "session_id": "test-session-1"},
        {"query": "What are good sources of vitamin D?", "session_id": "test-session-2"},
    ],
)
def test_rag_invoke_endpoint(test_input, monkeypatch):
    """
    Test the public endpoint for the RAG chain.
    
    This test mocks the NutritionRAGChain to avoid actual API calls.
    """
    # Mock the NutritionRAGChain invoke method to return a predefined response
    def mock_invoke(self, input_data, config=None):
        return RAGOutput(
            query=input_data.query,
            response="This is a mock response for testing purposes.",
            sources=[],
            structured_data={},
            session_id=input_data.session_id or "test-session"
        )
    
    # Apply the monkeypatch to the NutritionRAGChain.invoke method
    from wise_nutrition.rag_chain import NutritionRAGChain
    monkeypatch.setattr(NutritionRAGChain, "invoke", mock_invoke)
    
    # Call the public endpoint with the test input
    response = client.post("/api/v1/nutrition_rag_chain/public", json=test_input)
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == test_input["query"]
    assert "response" in data
    assert "sources" in data
    assert "structured_data" in data
    assert data["session_id"] == test_input["session_id"] 