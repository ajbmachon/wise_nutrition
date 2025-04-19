"""
Models for recommendation storage, tagging, and categorization.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from uuid import uuid4, UUID

# --- Base Models --- #

class TagBase(BaseModel):
    """Base model for tags."""
    name: str = Field(description="Tag name", max_length=50, min_length=1)
    description: Optional[str] = Field(None, description="Tag description")

class TagCreate(TagBase):
    """Schema for tag creation."""
    pass

class Tag(TagBase):
    """Schema for tag response."""
    id: UUID = Field(description="Tag unique identifier")
    created_at: datetime = Field(description="When the tag was created")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "protein",
                "description": "Protein-related recommendations",
                "created_at": "2023-01-01T00:00:00"
            }]
        }
    }

class CategoryBase(BaseModel):
    """Base model for categories."""
    name: str = Field(description="Category name", max_length=50, min_length=1)
    description: Optional[str] = Field(None, description="Category description")

class CategoryCreate(CategoryBase):
    """Schema for category creation."""
    pass

class Category(CategoryBase):
    """Schema for category response."""
    id: UUID = Field(description="Category unique identifier")
    created_at: datetime = Field(description="When the category was created")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Vitamins",
                "description": "Vitamin-related recommendations",
                "created_at": "2023-01-01T00:00:00"
            }]
        }
    }

class RecommendationBase(BaseModel):
    """Base model for recommendations."""
    title: str = Field(description="Recommendation title", max_length=200)
    content: str = Field(description="Recommendation content")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used for the recommendation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RecommendationCreate(RecommendationBase):
    """Schema for recommendation creation."""
    tag_ids: Optional[List[UUID]] = Field(None, description="IDs of tags to associate with this recommendation")
    category_id: Optional[UUID] = Field(None, description="ID of the category for this recommendation")

class RecommendationUpdate(BaseModel):
    """Schema for recommendation updates."""
    title: Optional[str] = Field(None, description="Recommendation title", max_length=200)
    content: Optional[str] = Field(None, description="Recommendation content")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used for the recommendation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tag_ids: Optional[List[UUID]] = Field(None, description="IDs of tags to associate with this recommendation")
    category_id: Optional[UUID] = Field(None, description="ID of the category for this recommendation")

class Recommendation(RecommendationBase):
    """Schema for recommendation response."""
    id: UUID = Field(description="Recommendation unique identifier")
    user_id: UUID = Field(description="ID of the user who owns this recommendation")
    tags: List[Tag] = Field(default_factory=list, description="Associated tags")
    category: Optional[Category] = Field(None, description="Associated category")
    created_at: datetime = Field(description="When the recommendation was created")
    updated_at: datetime = Field(description="When the recommendation was last updated")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Vitamin D Recommendation",
                "content": "Increase vitamin D intake through fatty fish, egg yolks, and sunlight exposure.",
                "sources": [{
                    "id": 1,
                    "content_preview": "Vitamin D is essential for calcium absorption...",
                    "source": "nutrition_database",
                    "type": "vitamin"
                }],
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "tags": [{
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "vitamins",
                    "description": "Vitamin-related recommendations"
                }],
                "category": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "Nutrition",
                    "description": "General nutrition recommendations"
                },
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-02T00:00:00",
                "metadata": {
                    "importance": "high",
                    "source_query": "vitamin d benefits"
                }
            }]
        }
    }

# --- For Database Models --- #

class TagInDB(TagBase):
    """Database model for tags."""
    id: UUID = Field(default_factory=uuid4, description="Tag unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the tag was created")
    
    model_config = {"arbitrary_types_allowed": True}

class CategoryInDB(CategoryBase):
    """Database model for categories."""
    id: UUID = Field(default_factory=uuid4, description="Category unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the category was created")
    
    model_config = {"arbitrary_types_allowed": True}

class RecommendationInDB(RecommendationBase):
    """Database model for recommendations."""
    id: UUID = Field(default_factory=uuid4, description="Recommendation unique identifier")
    user_id: UUID = Field(description="ID of the user who owns this recommendation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the recommendation was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the recommendation was last updated")
    
    # We'll store relationships through IDs in the database
    tag_ids: List[UUID] = Field(default_factory=list, description="IDs of associated tags")
    category_id: Optional[UUID] = Field(None, description="ID of the associated category")
    
    model_config = {"arbitrary_types_allowed": True}

# --- Export Schemas --- #

class ExportFormat(BaseModel):
    """Model for export format options."""
    format: str = Field(description="Export format (pdf or text)") 