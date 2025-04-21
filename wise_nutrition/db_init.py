"""Database initialization for Wise Nutrition."""

import logging
import os
from typing import Optional, Union, Any

# Use try/except for imports to provide better error messages
try:
    from supabase.client import create_client, Client
except ImportError:
    raise ImportError(
        "Could not import supabase. Make sure it's installed correctly with: \n"
        "pip install supabase==2.3.0 or poetry add supabase==2.3.0"
    )

try:
    import vecs
except ImportError:
    raise ImportError(
        "Could not import vecs. Make sure it's installed correctly with: \n"
        "pip install vecs==0.4.0 or poetry add vecs==0.4.0"
    )

from wise_nutrition.config import Config

# Set up logging
logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """Get the Supabase client instance.
    
    Following the official Supabase Python client documentation.
    
    Returns:
        Supabase client
    
    Raises:
        ValueError: If required environment variables are missing
    """
    url = Config.SUPABASE_URL
    key = Config.SUPABASE_KEY
    
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables are required"
        )
    
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {str(e)}")
        raise ValueError(f"Failed to create Supabase client: {str(e)}")


def get_vecs_client() -> Any:
    """Get the vecs client for pgvector operations.
    
    Returns:
        vecs client
        
    Raises:
        ValueError: If required environment variables are missing
    """
    connection_string = Config.POSTGRES_CONNECTION_STRING
    
    if not connection_string:
        raise ValueError(
            "POSTGRES_CONNECTION_STRING environment variable is required"
        )
    
    try:
        return vecs.create_client(connection_string)
    except Exception as e:
        logger.error(f"Failed to create vecs client: {str(e)}")
        raise ValueError(f"Failed to create vecs client: {str(e)}")


async def initialize_database() -> None:
    """Initialize the database with required extensions and tables.
    
    This function should be called once at application startup.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Check if pgvector is installed using SQL query
        try:
            response = supabase.table("pg_extension").select("*").eq("extname", "vector").execute()
            if response.data:
                logger.info("pgvector extension is installed")
            else:
                logger.warning(
                    "pgvector extension is not installed. "
                    "Please enable it in the Supabase dashboard."
                )
        except Exception as e:
            logger.warning(f"Could not check pgvector extension: {str(e)}")
            logger.warning("You may need to enable it manually in the Supabase dashboard.")
        
        # Initialize vecs client to verify connection
        try:
            vecs_client = get_vecs_client()
            logger.info("Vecs client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vecs client: {str(e)}")
        
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise


def create_exec_sql_function() -> None:
    """Create a PostgreSQL function to execute SQL statements.
    
    This function needs to be created manually in the Supabase dashboard
    with appropriate permissions.
    """
    sql = """
    CREATE OR REPLACE FUNCTION exec_sql(sql text)
    RETURNS SETOF record
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY EXECUTE sql;
    END;
    $$;
    """
    
    print(
        "Please execute the following SQL in the Supabase SQL Editor "
        "to create the exec_sql function:"
    )
    print(sql)
