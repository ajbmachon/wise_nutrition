"""
Router for the RAG chain endpoints using FastAPI dependencies.
"""
from typing import Annotated, Optional, Dict, List, Any, AsyncIterator
from fastapi import APIRouter, Depends, Request, Header, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from wise_nutrition.rag_chain import NutritionRAGChain, RAGInput, RAGOutput
from wise_nutrition.dependencies import get_rag_chain, get_retriever, get_llm, get_memory_manager, get_reranking_retriever, get_enhanced_reranking_retriever, get_citation_generator
from wise_nutrition.models.user import UserResponse
from wise_nutrition.auth.firebase_auth import get_current_active_user
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.documents import Document
from langserve import add_routes
import uuid

# Main router with prefix
router = APIRouter(prefix="/api/v1", tags=["RAG Chain"])

# Create a dedicated router for LangServe routes (without prefix)
langserve_router = APIRouter(tags=["RAG Chain"])

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
    try:
        # Directly invoke the chain with the input data
        result = rag_chain.invoke(input_data)
        return RAGOutput(**result)
    except Exception as e:
        print(f"Error in public endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

# Public streaming endpoint with limited functionality
@router.post("/nutrition_rag_chain/public/stream")
async def stream_rag_chain_public(
    input_data: RAGInput,
    rag_chain: Annotated[NutritionRAGChain, Depends(get_rag_chain)]
):
    """
    Public streaming endpoint for the RAG chain without authentication.
    Provides real-time responses with possible rate limits.
    """
    try:
        # Configure the run for streaming events
        config = RunnableConfig()

        # Define the streaming function using astream_events
        async def stream_response() -> AsyncIterator[str]:
            yield '{{"streaming": true, "session_id": "{}", "response_chunks": ['.format(input_data.session_id or str(uuid.uuid4()))
            
            first_chunk = True
            final_response = None
            
            # Use astream_events to get structured events
            async for event in rag_chain.astream_events(input_data, config=config, version="v1"):
                kind = event["event"]
                
                # Stream tokens from the LLM/ChatModel
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Escape quotes and newlines for JSON
                        content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        if first_chunk:
                            yield f'"{content}"'
                            first_chunk = False
                        else:
                            yield f', "{content}"'
                
                # Capture the final response when the full chain finishes
                elif kind == "on_chain_end":
                    if event["name"] == "NutritionRAGChain": # Check for the end of the main chain
                        final_response = event["data"].get("output")

            # Add final response data at the end
            # Ensure final_response structure is accessed correctly if it's not a simple dict
            query = input_data.query # Use the original input query
            response_text = ""
            if isinstance(final_response, dict):
                response_text = final_response.get("response", "")
            elif isinstance(final_response, str): # Handle cases where output might be a string
                response_text = final_response
            
            # Escape final response text
            response_text = response_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            
            yield f'], "query": "{query}"'
            yield f', "complete_response": "{response_text}"'
            yield '}'
        
        # Return a streaming response
        return StreamingResponse(
            stream_response(),
            media_type="application/json"
        )
    except Exception as e:
        print(f"Error in public streaming endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

# Custom endpoint to handle both LangServe and direct API formats
@router.post("/nutrition_rag_chain/invoke")
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

# Streaming endpoint for real-time chat responses
@router.post("/nutrition_rag_chain/stream")
async def stream_rag_chain(
    input_data: RAGInput,
    rag_chain: Annotated[NutritionRAGChain, Depends(get_rag_chain)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] # Add authentication dependency
):
    """
    Stream responses from the nutrition RAG chain for real-time chat experience.
    Requires authentication.

    Returns a streaming response with chunks of the generated text as they become available.
    """
    try:
        # Configure the run for streaming events
        config = RunnableConfig()

        # Define the streaming function using astream_events
        async def stream_response() -> AsyncIterator[str]:
            yield '{{"streaming": true, "session_id": "{}", "response_chunks": ['.format(input_data.session_id or str(uuid.uuid4()))
            
            first_chunk = True
            final_response = None

            # Use astream_events to get structured events
            async for event in rag_chain.astream_events(input_data, config=config, version="v1"):
                kind = event["event"]
                
                # Stream tokens from the LLM/ChatModel
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Escape quotes and newlines for JSON
                        content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        if first_chunk:
                            yield f'"{content}"'
                            first_chunk = False
                        else:
                            yield f', "{content}"'
                
                # Capture the final response when the full chain finishes
                elif kind == "on_chain_end":
                     if event["name"] == "NutritionRAGChain": # Check for the end of the main chain
                        final_response = event["data"].get("output")
            
            # Add final response data at the end
            query = input_data.query # Use the original input query
            response_text = ""
            if isinstance(final_response, dict):
                response_text = final_response.get("response", "")
            elif isinstance(final_response, str):
                response_text = final_response
            
            # Escape final response text
            response_text = response_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            yield f'], "query": "{query}"'
            yield f', "complete_response": "{response_text}"'
            yield '}'
        
        # Return a streaming response
        return StreamingResponse(
            stream_response(),
            media_type="application/json"
        )
    except Exception as e:
        print(f"Error in streaming endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

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

@router.get("/nutrition_rag_chain/health")
async def health_check():
    """
    Health check endpoint for the nutrition RAG chain.
    """
    return {"status": "ok", "message": "Nutrition RAG chain is ready"}

# --- Add a basic comparison endpoint to demonstrate the difference with reranking --- #
class RetrievalRequest(BaseModel):
    """Request model for comparing retrieval approaches."""
    query: str

class DocumentInfo(BaseModel):
    """Information about a retrieved document."""
    content: str
    metadata: Dict[str, Any]

class RetrievalResponse(BaseModel):
    """Response model for comparing retrieval approaches."""
    query: str
    standard_results: List[DocumentInfo]
    reranked_results: List[DocumentInfo]

@router.post("/compare_retrieval", response_model=RetrievalResponse)
async def compare_retrieval(
    request: RetrievalRequest,
    standard_retriever: Runnable = Depends(get_retriever),
    reranking_retriever: Runnable = Depends(get_enhanced_reranking_retriever)
):
    """
    Compare standard retrieval with reranked retrieval results.
    
    This endpoint demonstrates the difference between regular retrieval
    and retrieval with post-processing reranking to improve relevance.
    
    Args:
        request: The retrieval request containing the query
        
    Returns:
        A comparison of retrieval results with and without reranking
    """
    query = request.query
    
    # Get results from standard retriever
    standard_docs = standard_retriever.invoke(query)
    
    # Get results from reranking retriever
    reranked_docs = reranking_retriever.invoke(query)
    
    # Convert Document objects to DocumentInfo models
    standard_results = [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata
        ) for doc in standard_docs
    ]
    
    reranked_results = [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata
        ) for doc in reranked_docs
    ]
    
    return RetrievalResponse(
        query=query,
        standard_results=standard_results,
        reranked_results=reranked_results
    )

# Import dependencies instead of creating an instance directly
from wise_nutrition.dependencies import (
    get_retriever, 
    get_llm, 
    get_memory_manager, 
    get_citation_generator
)

# Create a chain instance to use with LangServe
rag_chain_instance = get_rag_chain(
    retriever=get_retriever(),
    llm=get_llm(),
    memory_manager=get_memory_manager(),
    citation_generator=get_citation_generator()
)

# Add LangServe routes for invoke, stream, and batch endpoints
add_routes(
    langserve_router,  # Use the router without prefix
    rag_chain_instance,  # Pass the instance, not the class
    path="/api/v1/nutrition_rag_chain",  # Include the full path
    include_callback_events=True  # For better debugging
)

# Include the LangServe router in the main router
router.include_router(langserve_router) 