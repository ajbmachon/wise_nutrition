"""Configuration module for Wise Nutrition."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # OpenAI API key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Supabase configuration
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_PROJECT_ID: str = os.getenv("SUPABASE_PROJECT_ID", "")
    SUPABASE_ACCESS_TOKEN: str = os.getenv("SUPABASE_ACCESS_TOKEN", "")
    
    # Database connection string for vecs
    POSTGRES_CONNECTION_STRING: Optional[str] = os.getenv("POSTGRES_CONNECTION_STRING", None)
    
    # Data directory
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
    
    # Vector dimensions for embeddings
    EMBEDDING_DIMENSIONS: int = 1536  # Default for OpenAI text-embedding-3-small
    
    @classmethod
    def get_postgres_connection(cls) -> str:
        """Get the PostgreSQL connection string.
        
        If POSTGRES_CONNECTION_STRING is provided, use that.
        Otherwise, construct it from Supabase credentials.
        
        Returns:
            PostgreSQL connection string
        """
        if cls.POSTGRES_CONNECTION_STRING:
            return cls.POSTGRES_CONNECTION_STRING
            
        # If no explicit connection string is provided, we need to construct one
        # This assumes you have direct database access credentials
        raise ValueError(
            "POSTGRES_CONNECTION_STRING environment variable is required "
            "for vector database operations"
        )
