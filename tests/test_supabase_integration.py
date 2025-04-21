"""Tests for Supabase integration."""

import os
import pytest
from dotenv import load_dotenv

from wise_nutrition.db_init import get_supabase_client, get_vecs_client
from wise_nutrition.config import Config

# Load environment variables
load_dotenv()


def test_supabase_client_creation():
    """Test that the Supabase client can be created successfully."""
    # Skip test if environment variables are not set
    if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
        pytest.skip("Supabase credentials not set in environment variables")
    
    # Create client
    client = get_supabase_client()
    
    # Verify client was created
    assert client is not None
    
    # Test a simple query to verify connection
    try:
        response = client.table("pg_extension").select("extname").execute()
        # Just checking if we get a response without errors
        assert hasattr(response, "data")
    except Exception as e:
        pytest.fail(f"Failed to execute query: {str(e)}")


def test_vecs_client_creation():
    """Test that the vecs client can be created successfully."""
    # Skip test if environment variable is not set
    if not Config.POSTGRES_CONNECTION_STRING:
        pytest.skip("Postgres connection string not set in environment variables")
    
    # Create client
    client = get_vecs_client()
    
    # Verify client was created
    assert client is not None
    
    # Test creating a collection (will be no-op if it already exists)
    try:
        collection = client.get_or_create_collection(
            name="test_collection", 
            dimension=Config.EMBEDDING_DIMENSIONS
        )
        assert collection is not None
    except Exception as e:
        pytest.fail(f"Failed to create collection: {str(e)}")


def test_vector_store_initialization():
    """Test that the vector store can be initialized."""
    # Skip test if environment variables are not set
    if not Config.POSTGRES_CONNECTION_STRING:
        pytest.skip("Postgres connection string not set in environment variables")
    
    # Import here to avoid circular imports
    from wise_nutrition.vector_store import SupabaseVectorStore
    
    try:
        # Create vector store
        vector_store = SupabaseVectorStore()
        
        # Verify collections were created
        assert vector_store.nutrient_info is not None
        assert vector_store.theory is not None
        assert vector_store.recipes is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize vector store: {str(e)}")
