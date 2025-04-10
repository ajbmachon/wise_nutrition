"""
Router for the RAG chain endpoints using FastAPI dependencies.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Request 
from pydantic import BaseModel

from wise_nutrition.rag_chain import NutritionRAGChain, RAGInput, RAGOutput
from wise_nutrition.dependencies import get_rag_chain

router = APIRouter()

@router.post("/nutrition_rag_chain/invoke", response_model=RAGOutput)
async def invoke_rag_chain(
    input_data: RAGInput,
    rag_chain: Annotated[NutritionRAGChain, Depends(get_rag_chain)]
):
    """
    Invoke the nutrition RAG chain to get a response based on the user query.
    """
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
            "invoke": "/api/v1/nutrition_rag_chain/invoke",
            "playground": "/api/v1/nutrition_rag_chain/playground"
        },
        "input_schema": {"$ref": "#/components/schemas/RAGInput"},
        "output_schema": {"$ref": "#/components/schemas/RAGOutput"}
    } 