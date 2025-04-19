"""
Router module exports.
"""

from wise_nutrition.routers.health import router as health_router
from wise_nutrition.routers.rag import router as rag_router
from wise_nutrition.routers.auth import router as auth_router
from wise_nutrition.routers.recommendations import router as recommendations_router
from wise_nutrition.routers.query_reformulation import router as query_reformulation_router

# Export the routers
health = health_router
rag = rag_router
auth = auth_router
recommendations = recommendations_router
query_reformulation = query_reformulation_router

__all__ = ["health", "rag", "auth", "recommendations", "query_reformulation"] 