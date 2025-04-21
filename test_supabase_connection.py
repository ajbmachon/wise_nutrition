"""
Simple script to test Supabase connection.
Run this script to verify that Supabase is correctly configured.
"""

import logging
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_connection():
    """Test the Supabase connection."""
    logger.info("Testing Supabase connection...")
    
    try:
        # Import here to catch import errors
        from wise_nutrition.db_init import get_supabase_client
        from wise_nutrition.config import Config
        
        # Check if environment variables are set
        if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
            logger.error("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
            return False
            
        # Create Supabase client
        logger.info("Creating Supabase client...")
        client = get_supabase_client()
        
        # Test a simple query
        logger.info("Executing test query...")
        response = client.table("pg_extension").select("extname").execute()
        
        logger.info(f"Query successful! Found {len(response.data)} extensions")
        for ext in response.data:
            logger.info(f"Extension: {ext.get('extname', 'unknown')}")
            
        # Test pgvector extension
        pgvector_installed = any(ext.get('extname') == 'vector' for ext in response.data)
        if pgvector_installed:
            logger.info("✅ pgvector extension is installed")
        else:
            logger.warning("⚠️ pgvector extension is not installed")
            
        logger.info("✅ Supabase connection test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Make sure you have installed the required packages:")
        logger.error("  poetry install")
        return False
    except Exception as e:
        logger.error(f"Error testing Supabase connection: {str(e)}")
        return False

def test_vecs_connection():
    """Test the vecs connection."""
    logger.info("Testing vecs connection...")
    
    try:
        # Import here to catch import errors
        from wise_nutrition.db_init import get_vecs_client
        from wise_nutrition.config import Config
        
        # Check if environment variable is set
        if not Config.POSTGRES_CONNECTION_STRING:
            logger.error("POSTGRES_CONNECTION_STRING environment variable must be set")
            return False
            
        # Create vecs client
        logger.info("Creating vecs client...")
        client = get_vecs_client()
        
        # Test creating a collection
        logger.info("Creating test collection...")
        collection = client.get_or_create_collection(
            name="test_collection", 
            dimension=Config.EMBEDDING_DIMENSIONS
        )
        
        logger.info(f"✅ Collection created: {collection.name}")
        logger.info("✅ Vecs connection test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Make sure you have installed the required packages:")
        logger.error("  poetry install")
        return False
    except Exception as e:
        logger.error(f"Error testing vecs connection: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=== Supabase Connection Test ===")
    
    # Test Supabase connection
    supabase_success = test_connection()
    
    # Test vecs connection
    vecs_success = test_vecs_connection()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Supabase connection: {'✅ SUCCESS' if supabase_success else '❌ FAILED'}")
    logger.info(f"Vecs connection: {'✅ SUCCESS' if vecs_success else '❌ FAILED'}")
    
    # Exit with appropriate status code
    sys.exit(0 if supabase_success and vecs_success else 1)
