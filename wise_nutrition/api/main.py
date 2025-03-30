"""
FastAPI main application.
"""
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from wise_nutrition.document_loaders.pdf_loader import NutritionPDFLoader
from wise_nutrition.embeddings.embedding_manager import EmbeddingManager
from wise_nutrition.memory.conversation_memory import ConversationMemoryManager
from wise_nutrition.rag.rag_chain import NutritionRAGChain
from wise_nutrition.retriever.custom_retriever import NutritionRetriever
from langgraph.checkpoint.memory import MemorySaver

# Global resources
# TODO: Update initialization parameters
memory_saver = MemorySaver()
conversation_manager = ConversationMemoryManager(memory_saver=memory_saver)
rag_chain = NutritionRAGChain()

app = FastAPI(
    title="Wise Nutrition API",
    description="A RAG-based nutrition advisor API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[str]
    session_id: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Wise Nutrition API"}


# TODO: This is a restricted admin endpoint. We need to add authentication. Perhaps move it to a separate admin API.
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be processed and indexed.
    """
    pass


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the nutrition advisor.
    
    This endpoint uses LangGraph for memory management.
    """
    # Get or create a session_id
    session_id = request.session_id or conversation_manager.generate_thread_id()
    
    # Use the rag_chain with the session_id
    response = rag_chain.invoke(
        query=request.query,
        session_id=session_id
    )
    
    # Add the session_id to the response
    response["session_id"] = session_id
    
    return response


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a conversation session.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    conversation_manager.clear(session_id)
    return {"message": f"Session {session_id} cleared successfully"}

# TODO: this is deprecated update to modern syntax
# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on startup.
    """
    global memory_saver, conversation_manager, rag_chain
    
    # Initialize the memory saver and conversation manager
    memory_saver = MemorySaver()
    conversation_manager = ConversationMemoryManager(memory_saver=memory_saver)
    
    # Other initializations would go here
    # (embedding_manager, retriever, rag_chain)


# TODO: this is deprecated update to modern syntax
# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources on shutdown.
    """
    # Any cleanup needed for langgraph memory saver
    pass 