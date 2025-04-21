"""Data models for Wise Nutrition."""

from typing import List, Optional
from pydantic import BaseModel, Field


class NutrientMetadata(BaseModel):
    """Metadata for nutrient information."""
    
    nutrient: str = Field(..., description="Name of the nutrient (e.g., 'Vitamin A')")
    category: str = Field(..., description="Category of information (e.g., 'symptoms', 'sources')")
    source: Optional[str] = Field(None, description="Reference source")
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the chunk")
    origin: Optional[str] = Field(None, description="Origin of the data (e.g., 'manual')")


class NutrientInfo(BaseModel):
    """Information about a nutrient."""
    
    text: str = Field(..., description="Nutrient information text")
    metadata: NutrientMetadata = Field(..., description="Metadata about the nutrient")


class RecipeIngredient(BaseModel):
    """Recipe ingredient."""
    
    name: str = Field(..., description="Ingredient name")
    amount: Optional[str] = Field(None, description="Ingredient quantity")
    notes: Optional[str] = Field(None, description="Optional notes about the ingredient")


class Recipe(BaseModel):
    """Recipe data model."""
    
    name: str = Field(..., description="Recipe name")
    description: str = Field(..., description="Recipe description")
    ingredients: List[str] = Field(..., description="List of ingredients")
    instructions: str = Field(..., description="Cooking instructions")
    prep_time: Optional[str] = Field(None, description="Preparation time")
    cook_time: Optional[str] = Field(None, description="Cooking time")
    fermentation_time: Optional[str] = Field(None, description="Fermentation time if applicable")
    nutrition_benefits: List[str] = Field(..., description="Nutritional benefits")
    tags: List[str] = Field(..., description="Recipe tags")


class NutritionTheory(BaseModel):
    """Nutrition theory information."""
    
    section: str = Field(..., description="Section name from the book")
    quote: str = Field(..., description="Original text from the book")
    summary: str = Field(..., description="Brief description of the content")
    tags: List[str] = Field(..., description="List of relevant tags")
    source: str = Field(..., description="Page reference")
    chunk_id: str = Field(..., description="Unique identifier")


class UserQuery(BaseModel):
    """User query model."""
    
    query: str = Field(..., description="User's natural language query")
    context: Optional[dict] = Field(None, description="Any additional context about the user")


class QueryType(BaseModel):
    """Query classification."""
    
    is_nutrition_question: bool = Field(..., description="Is this a question about nutrition?")
    is_recipe_request: bool = Field(..., description="Is this a request for recipes?")
    is_deficiency_question: bool = Field(..., description="Is this a question about deficiencies?")
    nutrients_mentioned: List[str] = Field(default_factory=list, description="Nutrients mentioned in the query")
    conditions_mentioned: List[str] = Field(default_factory=list, description="Health conditions mentioned")
    dietary_restrictions: List[str] = Field(default_factory=list, description="Dietary restrictions mentioned")
    query_summary: str = Field(..., description="Brief summary of the query intent")


class RetrievedContext(BaseModel):
    """Context retrieved from the vector store."""
    
    nutrition_info: List[NutrientInfo] = Field(default_factory=list, description="Retrieved nutrition information")
    recipes: List[Recipe] = Field(default_factory=list, description="Retrieved relevant recipes")
    theory: List[NutritionTheory] = Field(default_factory=list, description="Retrieved nutrition theory")
