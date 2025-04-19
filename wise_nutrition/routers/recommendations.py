"""
Router for recommendation operations including tagging, categorization, and export.
"""
import io
from typing import List, Dict, Any, Optional, Annotated
from uuid import UUID
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
    Body
)
from fastapi.responses import StreamingResponse

from wise_nutrition.models.recommendation import (
    RecommendationCreate,
    RecommendationUpdate,
    Recommendation,
    Tag,
    TagCreate,
    Category,
    CategoryCreate,
    ExportFormat
)
from wise_nutrition.models.user import UserResponse
from wise_nutrition.auth.firebase_auth import get_current_active_user
from wise_nutrition.storage.recommendation_storage import RecommendationStorageService
from wise_nutrition.storage.export_service import ExportService

# Create router
router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

# --- Dependency Injection --- #

def get_recommendation_storage() -> RecommendationStorageService:
    """Dependency for recommendation storage."""
    return RecommendationStorageService()

def get_export_service() -> ExportService:
    """Dependency for export service."""
    return ExportService()

# --- Tag Endpoints --- #

@router.post("/tags", response_model=Tag, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Create a new tag.
    
    Tags are shared across all users and can be used to categorize recommendations.
    """
    return await storage.create_tag(tag_data)

@router.get("/tags", response_model=List[Tag])
async def get_tags(
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Get all available tags.
    """
    return await storage.get_tags()

@router.get("/tags/{tag_id}", response_model=Tag)
async def get_tag(
    tag_id: UUID,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Get a tag by ID.
    """
    tag = await storage.get_tag(tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return tag

@router.put("/tags/{tag_id}", response_model=Tag)
async def update_tag(
    tag_id: UUID,
    tag_data: TagCreate,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Update a tag.
    """
    tag = await storage.update_tag(tag_id, tag_data)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return tag

@router.delete("/tags/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tag(
    tag_id: UUID,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Delete a tag.
    
    This will also remove the tag from all recommendations.
    """
    success = await storage.delete_tag(tag_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# --- Category Endpoints --- #

@router.post("/categories", response_model=Category, status_code=status.HTTP_201_CREATED)
async def create_category(
    category_data: CategoryCreate,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Create a new category.
    
    Categories are shared across all users and provide a high-level organization for recommendations.
    """
    return await storage.create_category(category_data)

@router.get("/categories", response_model=List[Category])
async def get_categories(
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Get all available categories.
    """
    return await storage.get_categories()

@router.get("/categories/{category_id}", response_model=Category)
async def get_category(
    category_id: UUID,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Get a category by ID.
    """
    category = await storage.get_category(category_id)
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    return category

@router.put("/categories/{category_id}", response_model=Category)
async def update_category(
    category_id: UUID,
    category_data: CategoryCreate,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Update a category.
    """
    category = await storage.update_category(category_id, category_data)
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    return category

@router.delete("/categories/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_category(
    category_id: UUID,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Delete a category.
    
    This will also remove the category from all recommendations.
    """
    success = await storage.delete_category(category_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# --- Recommendation Endpoints --- #

@router.post("", response_model=Recommendation, status_code=status.HTTP_201_CREATED)
async def create_recommendation(
    recommendation_data: RecommendationCreate,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Create a new recommendation.
    """
    return await storage.create_recommendation(current_user.id, recommendation_data)

@router.get("", response_model=List[Recommendation])
async def get_recommendations(
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    tag_ids: Optional[List[UUID]] = Query(None, description="Filter by tag IDs"),
    category_id: Optional[UUID] = Query(None, description="Filter by category ID"),
    search: Optional[str] = Query(None, description="Search in title and content"),
    limit: int = Query(100, description="Maximum number of results", gt=0, le=1000),
    offset: int = Query(0, description="Results offset for pagination", ge=0)
):
    """
    Get recommendations with optional filtering.
    """
    return await storage.get_recommendations(
        user_id=current_user.id,
        tag_ids=tag_ids,
        category_id=category_id,
        search_query=search,
        limit=limit,
        offset=offset
    )

@router.get("/{recommendation_id}", response_model=Recommendation)
async def get_recommendation(
    recommendation_id: UUID,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Get a recommendation by ID.
    """
    recommendation = await storage.get_recommendation(recommendation_id, current_user.id)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )
    return recommendation

@router.put("/{recommendation_id}", response_model=Recommendation)
async def update_recommendation(
    recommendation_id: UUID,
    recommendation_data: RecommendationUpdate,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Update a recommendation.
    """
    recommendation = await storage.update_recommendation(
        recommendation_id,
        current_user.id,
        recommendation_data
    )
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found or access denied"
        )
    return recommendation

@router.delete("/{recommendation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recommendation(
    recommendation_id: UUID,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Delete a recommendation.
    """
    success = await storage.delete_recommendation(recommendation_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found or access denied"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# --- Export Endpoints --- #

@router.post("/{recommendation_id}/export")
async def export_recommendation(
    recommendation_id: UUID,
    format_data: ExportFormat,
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    export_service: Annotated[ExportService, Depends(get_export_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Export a recommendation to the specified format (pdf or text).
    """
    # Get the recommendation
    recommendation = await storage.get_recommendation(recommendation_id, current_user.id)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )
    
    # Export based on format
    export_format = format_data.format.lower()
    if export_format == "pdf":
        try:
            content = await export_service.export_to_pdf(recommendation)
            filename = f"recommendation_{recommendation_id}.pdf"
            return StreamingResponse(
                io.BytesIO(content),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(e)
            )
    elif export_format == "text":
        content = await export_service.export_to_text(recommendation)
        filename = f"recommendation_{recommendation_id}.txt"
        return StreamingResponse(
            io.StringIO(content),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {export_format}. Supported formats: pdf, text"
        )

@router.post("/export")
async def export_recommendations(
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    export_service: Annotated[ExportService, Depends(get_export_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    format_data: ExportFormat,
    recommendation_ids: List[UUID] = Body(..., description="List of recommendation IDs to export")
):
    """
    Export multiple recommendations to the specified format (pdf or text).
    """
    # Get recommendations
    recommendations = []
    for rec_id in recommendation_ids:
        recommendation = await storage.get_recommendation(rec_id, current_user.id)
        if recommendation:
            recommendations.append(recommendation)
    
    if not recommendations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid recommendations found"
        )
    
    # Export based on format
    export_format = format_data.format.lower()
    if export_format == "pdf":
        try:
            content = await export_service.export_multiple_to_pdf(recommendations)
            filename = "recommendations.pdf"
            return StreamingResponse(
                io.BytesIO(content),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(e)
            )
    elif export_format == "text":
        content = await export_service.export_multiple_to_text(recommendations)
        filename = "recommendations.txt"
        return StreamingResponse(
            io.StringIO(content),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {export_format}. Supported formats: pdf, text"
        )

# --- Integration with RAG Chain --- #

@router.post("/from-response", response_model=Recommendation)
async def create_from_response(
    storage: Annotated[RecommendationStorageService, Depends(get_recommendation_storage)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    source_data: Dict[str, Any] = Body(..., description="Source response data from the RAG chain"),
    tag_ids: List[UUID] = Body(None, description="Optional tag IDs to apply"),
    category_id: Optional[UUID] = Body(None, description="Optional category ID to apply")
):
    """
    Create a recommendation from a RAG chain response.
    
    This endpoint allows saving a response from the nutrition RAG chain as a recommendation.
    """
    # Validate required fields are present
    if "query" not in source_data or "response" not in source_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: query and response are required"
        )
    
    # Create recommendation data
    recommendation_data = RecommendationCreate(
        title=source_data["query"][:100],  # Use truncated query as title
        content=source_data["response"],
        sources=source_data.get("sources", []),
        metadata={
            "source_query": source_data["query"],
            "session_id": source_data.get("session_id"),
            "created_from": "rag_response",
            "structured_data": source_data.get("structured_data", {})
        },
        tag_ids=tag_ids,
        category_id=category_id
    )
    
    # Create and return the recommendation
    return await storage.create_recommendation(current_user.id, recommendation_data) 