"""Nutrition Agent using Pydantic AI."""

from typing import List, Optional, Dict, Any, Union, Literal
import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from wise_nutrition.models import (
    UserQuery,
    QueryType,
    RetrievedContext,
    Recipe
)
from wise_nutrition.retrieval import RetrievalService, SimpleFileVectorStore


class NutrientRichFoods(BaseModel):
    """Foods rich in specific nutrients."""
    
    nutrient: str = Field(..., description="Nutrient name")
    foods: List[str] = Field(..., description="List of foods rich in this nutrient")
    daily_needs: Optional[str] = Field(None, description="Approximate daily needs if known")


class NutritionResponse(BaseModel):
    """Response to a nutrition query."""
    
    answer: str = Field(..., description="Detailed answer to the user's question")
    sources: List[str] = Field(default_factory=list, description="Sources used for the answer")
    nutrient_rich_foods: Optional[List[NutrientRichFoods]] = Field(
        None, 
        description="Foods rich in mentioned nutrients"
    )
    recipe_recommendations: Optional[List[str]] = Field(
        None,
        description="Names of recommended recipes if applicable"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )


class RecipeResponse(BaseModel):
    """Response containing recipe recommendations."""
    
    introduction: str = Field(
        ..., 
        description="Introduction explaining the recipe recommendations"
    )
    recipes: List[Recipe] = Field(..., description="Recommended recipes")
    nutrition_notes: str = Field(
        ..., 
        description="Notes about nutritional benefits of these recipes"
    )
    preparation_tips: Optional[str] = Field(
        None,
        description="Additional tips for preparation"
    )


class WiseNutritionResponse(BaseModel):
    """Combined response model for the Wise Nutrition agent."""
    
    response_type: Literal["nutrition", "recipe"] = Field(
        ...,
        description="Type of response provided"
    )
    nutrition_response: Optional[NutritionResponse] = Field(
        None,
        description="Nutrition information response"
    )
    recipe_response: Optional[RecipeResponse] = Field(
        None,
        description="Recipe recommendations response"
    )


class NutritionAgent:
    """Wise Nutrition Agent for answering nutrition queries and recommending recipes."""
    
    def __init__(
        self, 
        data_dir: Union[str, Path] = None,
        openai_api_key: Optional[str] = None
    ):
        """Initialize the nutrition agent.
        
        Args:
            data_dir: Directory containing nutrition data files
            openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
        """
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Default to data directory in project root
        if data_dir is None:
            # Assume data dir is at the same level as the wise_nutrition package
            current_dir = Path(__file__).parent
            data_dir = current_dir.parent / "data"
        
        # Initialize vector store and retrieval service
        self.vector_store = SimpleFileVectorStore(data_dir)
        self.retrieval_service = RetrievalService(self.vector_store)
        
        # Define system prompts
        self.query_analysis_prompt = """
        You are an expert nutritionist specializing in whole food nutrition based on Weston A. Price principles.
        Your task is to analyze the user's query and classify it accurately to determine the best way to respond.
        """
        
        self.nutrition_prompt = """
        You are a nutrition assistant specializing in whole food nutrition based on Weston A. Price principles.
        
        Key principles to follow in your responses:
        1. Always prioritize whole foods over supplements
        2. Emphasize traditional food preparation methods (fermentation, soaking, sprouting)
        3. Recognize the importance of animal foods and properly prepared plant foods
        4. Acknowledge the role of fat-soluble vitamins (A, D, K2) in nutrition
        5. Provide science-based information with appropriate caveats when evidence is limited
        
        When answering questions:
        - Provide practical, actionable advice based on whole foods
        - Explain the mechanism of action when relevant
        - Be honest about limitations in nutritional science
        - Cite sources for your information when available
        - Recommend specific foods rich in nutrients mentioned in the query
        
        Use the provided context to answer questions accurately. If the context doesn't contain
        enough information to fully answer the question, acknowledge the limitations while
        providing the best information available.
        """
        
        self.recipe_prompt = """
        You are a culinary nutrition expert specializing in traditional food preparation based
        on Weston A. Price principles.
        
        When recommending recipes:
        1. Focus on nutrient-density and traditional preparation methods
        2. Explain the health benefits of specific ingredients and preparation techniques
        3. Consider any dietary restrictions or preferences mentioned by the user
        4. Provide context about why certain foods and preparation methods are beneficial
        5. Include practical tips for preparation and storage
        
        Your recommendations should emphasize:
        - Properly prepared whole foods
        - Traditional fermentation when appropriate
        - Balanced nutrition with emphasis on fat-soluble vitamins
        - Seasonal and local ingredients when possible
        
        Use the provided recipe context to make specific recommendations, explaining why
        each recipe would be beneficial for the user's needs.
        """
        
        # Initialize agents
        self.query_analyzer = Agent(
            "openai:gpt-4o-mini",
            output_type=QueryType,
            system_prompt=self.query_analysis_prompt
        )
        
        self.nutrition_agent = Agent(
            "openai:gpt-4o",
            output_type=NutritionResponse,
            system_prompt=self.nutrition_prompt
        )
        
        self.recipe_agent = Agent(
            "openai:gpt-4o",
            output_type=RecipeResponse,
            system_prompt=self.recipe_prompt
        )
    
    async def analyze_query(self, query: str) -> QueryType:
        """Analyze the user's query to determine the type of information needed.
        
        Args:
            query: User's natural language query
            
        Returns:
            Query type analysis
        """
        result = await self.query_analyzer.run(query)
        return result.output
    
    async def answer_question(
        self, 
        query: str,
        context: RetrievedContext,
        query_analysis: QueryType
    ) -> NutritionResponse:
        """Answer a nutrition question based on retrieved context.
        
        Args:
            query: User's query
            context: Retrieved context
            query_analysis: Query analysis results
            
        Returns:
            Nutrition response
        """
        # Prepare context for the agent
        nutrition_info = [item.dict() for item in context.nutrition_info]
        theory_info = [item.dict() for item in context.theory]
        
        context_prompt = f"""
        USER QUERY: {query}
        
        QUERY ANALYSIS:
        {query_analysis.dict()}
        
        RELEVANT NUTRITION INFORMATION:
        {nutrition_info}
        
        RELEVANT NUTRITION THEORY:
        {theory_info}
        """
        
        result = await self.nutrition_agent.run(context_prompt)
        return result.output
    
    async def recommend_recipes(
        self,
        query: str,
        context: RetrievedContext,
        query_analysis: QueryType
    ) -> RecipeResponse:
        """Recommend recipes based on retrieved context.
        
        Args:
            query: User's query
            context: Retrieved context
            query_analysis: Query analysis results
            
        Returns:
            Recipe recommendations
        """
        # Prepare context for the agent
        recipes = [item.dict() for item in context.recipes]
        
        context_prompt = f"""
        USER QUERY: {query}
        
        QUERY ANALYSIS:
        {query_analysis.dict()}
        
        AVAILABLE RECIPES:
        {recipes}
        """
        
        result = await self.recipe_agent.run(context_prompt)
        return result.output
    
    async def process_query(self, query: str) -> WiseNutritionResponse:
        """Process a user query and generate a response.
        
        Args:
            query: User's natural language query
            
        Returns:
            Wise Nutrition response with either nutrition info or recipe recommendations
        """
        # Analyze the query to determine intent
        query_analysis = await self.analyze_query(query)
        
        # Retrieve relevant context
        context = await self.retrieval_service.retrieve_context(query)
        
        # Determine if this is primarily a nutrition question or recipe request
        if query_analysis.is_recipe_request:
            # User is looking for recipes
            recipe_response = await self.recommend_recipes(query, context, query_analysis)
            return WiseNutritionResponse(
                response_type="recipe",
                recipe_response=recipe_response
            )
        else:
            # User is asking a nutrition question
            nutrition_response = await self.answer_question(query, context, query_analysis)
            return WiseNutritionResponse(
                response_type="nutrition",
                nutrition_response=nutrition_response
            )
