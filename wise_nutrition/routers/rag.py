"""
Router for the RAG chain endpoints using FastAPI dependencies.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Request, Header, HTTPException, status
from pydantic import BaseModel

from wise_nutrition.rag_chain import NutritionRAGChain, RAGInput, RAGOutput
from wise_nutrition.dependencies import get_rag_chain
from wise_nutrition.models.user import UserResponse
from wise_nutrition.auth.firebase_auth import get_current_active_user

router = APIRouter()

# Public endpoint with limited functionality
@router.post("/nutrition_rag_chain/public", response_model=RAGOutput)
async def invoke_rag_chain_public(
    input_data: RAGInput,
    rag_chain: Annotated[NutritionRAGChain, Depends(get_rag_chain)]
):
    """
    Public endpoint to invoke the nutrition RAG chain without authentication.
    May have rate limits or reduced functionality compared to the authenticated version.
    """
    # Here you could add rate limiting, reduce context length, etc.
    result = rag_chain.invoke(input_data)
    return result

# Authenticated endpoint with full functionality
@router.post("/nutrition_rag_chain/invoke", response_model=RAGOutput)
async def invoke_rag_chain(
    input_data: RAGInput,
    rag_chain: Annotated[NutritionRAGChain, Depends(get_rag_chain)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Invoke the nutrition RAG chain to get a response based on the user query.
    Requires authentication.
    """
    # Use the authenticated user's ID as the session ID if none provided
    if not input_data.session_id:
        input_data.session_id = str(current_user.id)
    
    # Here you could add user-specific logic, like:
    # - Personalization based on user preferences
    # - Tracking user usage
    # - Access to premium features
    
    result = rag_chain.invoke(input_data)
    return result

@router.get("/nutrition_rag_chain/playground")
async def rag_chain_playground():
    """
    Playground HTML interface for the nutrition RAG chain.
    """
    return {
        "message": "Playground is not available with direct FastAPI endpoints. Try using /api/v1/nutrition_rag_chain/invoke with a POST request."
    }

@router.get("/nutrition_rag_chain")
async def rag_chain_info():
    """
    Get information about the nutrition RAG chain.
    """
    return {
        "name": "Nutrition RAG Chain",
        "description": "A RAG chain for nutrition-related queries",
        "endpoints": {
            "public": "/api/v1/nutrition_rag_chain/public",
            "authenticated": "/api/v1/nutrition_rag_chain/invoke",
            "playground": "/api/v1/nutrition_rag_chain/playground"
        },
        "input_schema": {"$ref": "#/components/schemas/RAGInput"},
        "output_schema": {"$ref": "#/components/schemas/RAGOutput"},
        "authentication": "Bearer token authentication required for /invoke endpoint"
    }

# Test endpoint for LangSmith tracing
@router.post("/nutrition_rag_chain/debug_trace", response_model=RAGOutput)
async def debug_trace(
    input_data: RAGInput,
    rag_chain: Annotated[NutritionRAGChain, Depends(get_rag_chain)]
):
    """
    Debug endpoint to test LangSmith tracing.
    This endpoint explicitly sets tracing configs.
    """
    from wise_nutrition.utils.config import Config
    config = Config()
    
    # Log the tracing configuration
    print(f"LangSmith tracing: {config.langsmith_tracing}")
    print(f"LangSmith project: {config.langsmith_project}")
    print(f"LangSmith endpoint: {config.langsmith_endpoint}")
    
    # Special config for debug tracing
    trace_config = {
        "configurable": {"session_id": "debug-session"},
        "tags": ["debug", "testing", "nutrition_rag"],
        "run_name": f"Debug Trace: {input_data.query[:30]}...",
        "callbacks": []
    }
    
    # Force tracing for this call
    if not config.langsmith_tracing:
        import os
        import langchain
        print("Enabling tracing for this request only")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "debug_traces"
    
    result = rag_chain.invoke(input_data, config=trace_config)
    return result 