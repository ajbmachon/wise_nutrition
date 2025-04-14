"""
FastAPI main application using APIRouters.
"""
import os
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from wise_nutrition.routers import health, rag, auth

# --- FastAPI App Setup --- #
app = FastAPI(
    title="Wise Nutrition API",
    description="A RAG-based nutrition advisor API with LangServe endpoints",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers --- #
app.include_router(health.router, tags=["Health"])
# Include the RAG router with a prefix
app.include_router(rag.router, prefix="/api/v1", tags=["RAG Chain"])
# Include the auth router with a prefix
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])

# --- Root Endpoint --- #
@app.get("/")
async def root():
    """Root endpoint providing basic info and link to docs/playground."""
    return {
        "message": "Welcome to the Wise Nutrition API.",
        "docs": "/docs",
        "rag_playground": "/api/v1/nutrition_rag_chain/playground",
        "auth": "/api/v1/auth"
    }

# --- Custom Exception Handlers (Example Placeholder) --- #
# You can add custom handlers to catch specific exceptions from your
# chain or dependencies and return standardized error responses.
# from fastapi import Request
# from fastapi.responses import JSONResponse
#
# class ChainInvocationError(Exception):
#     pass
#
# @app.exception_handler(ChainInvocationError)
# async def chain_invocation_exception_handler(request: Request, exc: ChainInvocationError):
#     return JSONResponse(
#         status_code=500,
#         content={"message": f"Error processing request: {exc}"},
#     )
#
# @app.exception_handler(ValueError) # Example for built-in errors
# async def value_error_exception_handler(request: Request, exc: ValueError):
#     return JSONResponse(
#         status_code=400, # Bad request
#         content={"message": f"Invalid input: {exc}"},
#     )

# --- Removed Old Logic --- #
# - RAG chain instantiation (moved to routers/rag.py and dependencies.py)
# - LangServe add_routes call (moved to routers/rag.py)
# - Health check endpoint (moved to routers/health.py)

# --- Run the app (for local development) --- #
if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development
    uvicorn.run("wise_nutrition.api:app", host="0.0.0.0", port=8002, reload=True) 