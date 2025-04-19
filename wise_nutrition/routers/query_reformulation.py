"""
Router for query reformulation endpoints.
"""
from typing import Dict, List, Any
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable 

from wise_nutrition.query_reformulation import QueryReformulator
from wise_nutrition.dependencies import get_llm, get_query_reformulator

router = APIRouter(prefix="/api/v1/query", tags=["Query Reformulation"])

class QueryRequest(BaseModel):
    """Request model for query reformulation."""
    query: str

class QueryResponse(BaseModel):
    """Response model for reformulated queries."""
    original_query: str
    reformulated_queries: List[str]

@router.post("/reformulate", response_model=QueryResponse)
async def reformulate_query(
    request: QueryRequest,
    query_reformulator: QueryReformulator = Depends(get_query_reformulator),
) -> QueryResponse:
    """
    Reformulate a user query to generate multiple perspectives for improved retrieval.
    
    Args:
        request: The query request containing the original query.
        query_reformulator: The query reformulation module.
        
    Returns:
        A list of reformulated queries.
    """
    try:
        reformulated_queries = query_reformulator.rewrite_query(request.query)
        return QueryResponse(
            original_query=request.query,
            reformulated_queries=reformulated_queries
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reformulating query: {str(e)}"
        ) 