"""Tests for the agent module."""

import json
import os
from pathlib import Path
import pytest
from typing import Dict, Any, List, Optional
import tempfile
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

from wise_nutrition.agent import (
    NutritionAgent, 
    QueryType, 
    RetrievedContext,
    NutritionResponse,
    RecipeResponse,
    WiseNutritionResponse
)
from wise_nutrition.models import NutrientInfo, Recipe, NutritionTheory


@pytest.fixture
def sample_query_result() -> QueryType:
    """Create a sample query analysis result.
    
    Returns:
        Sample query analysis
    """
    return QueryType(
        is_nutrition_question=True,
        is_recipe_request=False,
        is_deficiency_question=False,
        nutrients_mentioned=["Vitamin D"],
        conditions_mentioned=[],
        dietary_restrictions=[],
        query_summary="User is asking about vitamin D benefits"
    )


@pytest.fixture
def sample_retrieved_context() -> RetrievedContext:
    """Create a sample retrieved context.
    
    Returns:
        Sample retrieved context
    """
    return RetrievedContext(
        nutrition_info=[
            NutrientInfo(
                text="Vitamin D is produced in the skin when exposed to sunlight.",
                metadata={"nutrient": "Vitamin D", "category": "sources"}
            ),
            NutrientInfo(
                text="Vitamin D deficiency may cause bone pain and muscle weakness.",
                metadata={"nutrient": "Vitamin D", "category": "symptoms"}
            )
        ],
        recipes=[
            Recipe(
                name="Salmon with Lemon",
                description="Simple salmon dish rich in vitamin D and omega-3 fatty acids.",
                ingredients=["salmon", "lemon", "salt", "olive oil"],
                instructions="Bake salmon with lemon, salt, and olive oil.",
                nutrition_benefits=["Rich in vitamin D", "High in omega-3 fatty acids"],
                tags=["high-vitamin-D", "seafood", "omega-3"]
            )
        ],
        theory=[
            NutritionTheory(
                section="FAT-SOLUBLE VITAMINS",
                quote="Vitamin D is found only in animal fats.",
                summary="Information about vitamin D sources.",
                tags=["vitamin D", "animal fats"],
                source="Nourishing Traditions, p. 42",
                chunk_id="theory_fats_42_01"
            )
        ]
    )


@pytest.fixture
def sample_nutrition_response() -> NutritionResponse:
    """Create a sample nutrition response.
    
    Returns:
        Sample nutrition response
    """
    return NutritionResponse(
        answer="Vitamin D is essential for bone health and immune function. It can be obtained from sunlight exposure and animal fats.",
        sources=["Nourishing Traditions, p. 42"],
        nutrient_rich_foods=[
            {
                "nutrient": "Vitamin D",
                "foods": ["Salmon", "Egg yolks", "Liver", "Cod liver oil"],
                "daily_needs": "600-800 IU for adults"
            }
        ],
        recipe_recommendations=["Salmon with Lemon"],
        follow_up_questions=[
            "How can I get more vitamin D during winter?",
            "Are there any plant sources of vitamin D?"
        ]
    )


@pytest.fixture
def sample_recipe_response() -> RecipeResponse:
    """Create a sample recipe response.
    
    Returns:
        Sample recipe response
    """
    return RecipeResponse(
        introduction="Here are some recipes rich in vitamin D to support bone and immune health:",
        recipes=[
            Recipe(
                name="Salmon with Lemon",
                description="Simple salmon dish rich in vitamin D and omega-3 fatty acids.",
                ingredients=["salmon", "lemon", "salt", "olive oil"],
                instructions="Bake salmon with lemon, salt, and olive oil.",
                nutrition_benefits=["Rich in vitamin D", "High in omega-3 fatty acids"],
                tags=["high-vitamin-D", "seafood", "omega-3"]
            )
        ],
        nutrition_notes="These recipes provide natural sources of vitamin D, which is essential for calcium absorption and immune function.",
        preparation_tips="For maximum nutrient retention, avoid overcooking salmon."
    )


async def mock_query_analyzer_function(
    messages: List[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Mock function for query analyzer.
    
    Args:
        messages: Model messages
        info: Agent info
        
    Returns:
        Model response containing query analysis
    """
    return ModelResponse(parts=[TextPart(
        """{
            "is_nutrition_question": true,
            "is_recipe_request": false,
            "is_deficiency_question": false,
            "nutrients_mentioned": ["Vitamin D"],
            "conditions_mentioned": [],
            "dietary_restrictions": [],
            "query_summary": "User is asking about vitamin D benefits"
        }"""
    )])


async def mock_nutrition_agent_function(
    messages: List[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Mock function for nutrition agent.
    
    Args:
        messages: Model messages
        info: Agent info
        
    Returns:
        Model response containing nutrition response
    """
    return ModelResponse(parts=[TextPart(
        """{
            "answer": "Vitamin D is essential for bone health and immune function. It can be obtained from sunlight exposure and animal fats.",
            "sources": ["Nourishing Traditions, p. 42"],
            "nutrient_rich_foods": [
                {
                    "nutrient": "Vitamin D",
                    "foods": ["Salmon", "Egg yolks", "Liver", "Cod liver oil"],
                    "daily_needs": "600-800 IU for adults"
                }
            ],
            "recipe_recommendations": ["Salmon with Lemon"],
            "follow_up_questions": [
                "How can I get more vitamin D during winter?",
                "Are there any plant sources of vitamin D?"
            ]
        }"""
    )])


async def mock_recipe_agent_function(
    messages: List[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Mock function for recipe agent.
    
    Args:
        messages: Model messages
        info: Agent info
        
    Returns:
        Model response containing recipe response
    """
    return ModelResponse(parts=[TextPart(
        """{
            "introduction": "Here are some recipes rich in vitamin D to support bone and immune health:",
            "recipes": [
                {
                    "name": "Salmon with Lemon",
                    "description": "Simple salmon dish rich in vitamin D and omega-3 fatty acids.",
                    "ingredients": ["salmon", "lemon", "salt", "olive oil"],
                    "instructions": "Bake salmon with lemon, salt, and olive oil.",
                    "nutrition_benefits": ["Rich in vitamin D", "High in omega-3 fatty acids"],
                    "tags": ["high-vitamin-D", "seafood", "omega-3"]
                }
            ],
            "nutrition_notes": "These recipes provide natural sources of vitamin D, which is essential for calcium absorption and immune function.",
            "preparation_tips": "For maximum nutrient retention, avoid overcooking salmon."
        }"""
    )])


@pytest.mark.asyncio
async def test_analyze_query():
    """Test query analysis function with expected input."""
    # Create mock agent with function model
    with patch('wise_nutrition.agent.Agent') as MockAgent:
        # Configure the mock
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.output = QueryType(
            is_nutrition_question=True,
            is_recipe_request=False,
            is_deficiency_question=False,
            nutrients_mentioned=["Vitamin D"],
            conditions_mentioned=[],
            dietary_restrictions=[],
            query_summary="User is asking about vitamin D benefits"
        )
        
        # Create agent and test
        agent = NutritionAgent()
        result = await agent.analyze_query("What are the benefits of vitamin D?")
        
        # Verify results
        assert result.is_nutrition_question is True
        assert "Vitamin D" in result.nutrients_mentioned
        assert result.query_summary == "User is asking about vitamin D benefits"


@pytest.mark.asyncio
async def test_answer_question(sample_query_result, sample_retrieved_context):
    """Test answering a nutrition question with expected inputs.
    
    Args:
        sample_query_result: Sample query analysis
        sample_retrieved_context: Sample retrieved context
    """
    # Create mock agent with function model
    with patch('wise_nutrition.agent.Agent') as MockAgent:
        # Configure the mock
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.output = NutritionResponse(
            answer="Vitamin D is essential for bone health and immune function. It can be obtained from sunlight exposure and animal fats.",
            sources=["Nourishing Traditions, p. 42"],
            nutrient_rich_foods=[
                {
                    "nutrient": "Vitamin D",
                    "foods": ["Salmon", "Egg yolks", "Liver", "Cod liver oil"],
                    "daily_needs": "600-800 IU for adults"
                }
            ],
            recipe_recommendations=["Salmon with Lemon"],
            follow_up_questions=[
                "How can I get more vitamin D during winter?",
                "Are there any plant sources of vitamin D?"
            ]
        )
        
        # Create agent and test
        agent = NutritionAgent()
        result = await agent.answer_question(
            "What are the benefits of vitamin D?",
            sample_retrieved_context,
            sample_query_result
        )
        
        # Verify results
        assert "bone health" in result.answer
        assert len(result.sources) > 0
        assert len(result.nutrient_rich_foods) > 0
        assert result.nutrient_rich_foods[0]["nutrient"] == "Vitamin D"
        assert len(result.follow_up_questions) > 0


@pytest.mark.asyncio
async def test_process_query_nutrition_question():
    """Test processing a nutrition query (expected use case)."""
    # Mock all necessary components
    with patch('wise_nutrition.agent.Agent') as MockAgent:
        # Configure the mocks
        mock_agent_instance = MockAgent.return_value
        
        # Setup different outputs based on which agent is being called
        def mock_run(prompt):
            if "analyzing" in str(prompt).lower() or "classify" in str(prompt).lower():
                # For query analyzer
                mock_output = AsyncMock()
                mock_output.output = QueryType(
                    is_nutrition_question=True,
                    is_recipe_request=False,
                    is_deficiency_question=False,
                    nutrients_mentioned=["Vitamin D"],
                    conditions_mentioned=[],
                    dietary_restrictions=[],
                    query_summary="User is asking about vitamin D benefits"
                )
                return mock_output
            else:
                # For nutrition agent
                mock_output = AsyncMock()
                mock_output.output = NutritionResponse(
                    answer="Vitamin D is essential for bone health and immune function.",
                    sources=["Nourishing Traditions, p. 42"],
                    nutrient_rich_foods=[
                        {
                            "nutrient": "Vitamin D",
                            "foods": ["Salmon", "Egg yolks", "Liver"],
                            "daily_needs": "600-800 IU for adults"
                        }
                    ],
                    recipe_recommendations=["Salmon with Lemon"],
                    follow_up_questions=["How can I get more vitamin D during winter?"]
                )
                return mock_output
        
        mock_agent_instance.run.side_effect = mock_run
        
        # Mock retrieval service
        with patch('wise_nutrition.agent.RetrievalService') as MockRetrievalService:
            mock_retrieval_instance = MockRetrievalService.return_value
            mock_retrieval_instance.retrieve_context.return_value = RetrievedContext(
                nutrition_info=[
                    NutrientInfo(
                        text="Vitamin D is produced in the skin when exposed to sunlight.",
                        metadata={"nutrient": "Vitamin D", "category": "sources"}
                    )
                ],
                recipes=[],
                theory=[]
            )
            
            # Create agent and test
            agent = NutritionAgent()
            result = await agent.process_query("What are the benefits of vitamin D?")
            
            # Verify results
            assert result.response_type == "nutrition"
            assert result.nutrition_response is not None
            assert "bone health" in result.nutrition_response.answer
            assert "Vitamin D" in result.nutrition_response.nutrient_rich_foods[0]["nutrient"]


@pytest.mark.asyncio
async def test_process_query_empty_query():
    """Test processing an empty query (edge case)."""
    # Create agent with mock components
    with patch('wise_nutrition.agent.Agent') as MockAgent:
        # This test might raise an exception or return empty results
        # depending on implementation
        mock_agent_instance = MockAgent.return_value
        
        # Return error response for empty query
        mock_output = AsyncMock()
        mock_output.output = QueryType(
            is_nutrition_question=False,
            is_recipe_request=False,
            is_deficiency_question=False,
            nutrients_mentioned=[],
            conditions_mentioned=[],
            dietary_restrictions=[],
            query_summary="Empty or unclear query"
        )
        mock_agent_instance.run.return_value = mock_output
        
        # Mock retrieval service
        with patch('wise_nutrition.agent.RetrievalService') as MockRetrievalService:
            mock_retrieval_instance = MockRetrievalService.return_value
            mock_retrieval_instance.retrieve_context.return_value = RetrievedContext(
                nutrition_info=[],
                recipes=[],
                theory=[]
            )
            
            # Create agent and test
            agent = NutritionAgent()
            
            # Test with empty query
            # This should either return a minimal response or raise an exception
            # depending on the implementation
            try:
                result = await agent.process_query("")
                # If it returns, verify it has the expected structure
                assert result.response_type in ("nutrition", "recipe")
                if result.response_type == "nutrition":
                    assert result.nutrition_response is not None
                else:
                    assert result.recipe_response is not None
            except ValueError:
                # If it raises an exception, that's also acceptable for an edge case
                pass


@pytest.mark.asyncio
async def test_process_query_invalid_model(sample_query_result, sample_retrieved_context):
    """Test handling invalid model response (failure case).
    
    Args:
        sample_query_result: Sample query analysis
        sample_retrieved_context: Sample retrieved context
    """
    # Create agent with mock components
    with patch('wise_nutrition.agent.Agent') as MockAgent:
        # Configure the query analyzer mock
        mock_agent_instance = MockAgent.return_value
        
        # For query analyzer
        mock_output_query = AsyncMock()
        mock_output_query.output = sample_query_result
        
        # For nutrition agent - simulate an exception
        mock_agent_instance.run.side_effect = [
            mock_output_query,  # First call returns query analysis
            Exception("Model error"),  # Second call raises exception
        ]
        
        # Mock retrieval service
        with patch('wise_nutrition.agent.RetrievalService') as MockRetrievalService:
            mock_retrieval_instance = MockRetrievalService.return_value
            mock_retrieval_instance.retrieve_context.return_value = sample_retrieved_context
            
            # Create agent
            agent = NutritionAgent()
            
            # Test with a valid query but simulate a model error
            with pytest.raises(Exception):
                await agent.process_query("What are the benefits of vitamin D?")
