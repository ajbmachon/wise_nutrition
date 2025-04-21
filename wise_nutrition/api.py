"""FastAPI application for Wise Nutrition."""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel

from wise_nutrition.config import Config
from wise_nutrition.vector_store import SupabaseVectorStore
from wise_nutrition.retrieval import RetrievalService
from wise_nutrition.agent import NutritionAgent
from wise_nutrition.db_init import initialize_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Wise Nutrition API",
    description="AI-powered nutrition knowledge and recipe recommendations",
    version="0.1.0",
)


# Models for API requests and responses
class QueryRequest(BaseModel):
    """Query request model."""
    
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 5


class QueryResponse(BaseModel):
    """Query response model."""
    
    response: str
    sources: List[Dict[str, Any]] = []
    response_type: str = "nutrition"


# Dependency for vector store
async def get_vector_store() -> SupabaseVectorStore:
    """Get vector store instance.
    
    Returns:
        Vector store instance
    """
    try:
        # Use the connection string from config
        return SupabaseVectorStore()
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize vector store: {str(e)}"
        )


# Dependency for retrieval service
async def get_retrieval_service(
    vector_store: SupabaseVectorStore = Depends(get_vector_store)
) -> RetrievalService:
    """Get retrieval service instance.
    
    Args:
        vector_store: Vector store instance
        
    Returns:
        Retrieval service instance
    """
    return RetrievalService(vector_store)


# Dependency for nutrition agent
async def get_nutrition_agent(
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
) -> NutritionAgent:
    """Get nutrition agent instance.
    
    Args:
        retrieval_service: Retrieval service instance
        
    Returns:
        Nutrition agent instance
    """
    return NutritionAgent(retrieval_service=retrieval_service)


@app.on_event("startup")
async def startup_event():
    """Initialize database and other resources on startup."""
    try:
        await initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint.
    
    Returns:
        Welcome message
    """
    return {"message": "Welcome to Wise Nutrition API"}


@app.post("/api/ingest", status_code=200)
async def ingest_data(
    vector_store: SupabaseVectorStore = Depends(get_vector_store),
    data_dir: Optional[str] = None
):
    """Ingest data from JSON files into vector collections.
    
    Args:
        vector_store: Vector store instance
        data_dir: Optional path to data directory
        
    Returns:
        Status message
    """
    try:
        data_path = Path(data_dir) if data_dir else None
        vector_store.ingest_data(data_path)
        return {"status": "success", "message": "Data ingested successfully"}
    except Exception as e:
        logger.error(f"Error ingesting data: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest data: {str(e)}"
        )


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    agent: NutritionAgent = Depends(get_nutrition_agent)
):
    """Process a nutrition query.
    
    Args:
        request: Query request
        agent: Nutrition agent instance
        
    Returns:
        Query response
    """
    try:
        response = await agent.process_query(
            request.query, 
            filters=request.filters
        )
        
        return QueryResponse(
            response=response.response,
            sources=response.sources,
            response_type=response.response_type
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}
