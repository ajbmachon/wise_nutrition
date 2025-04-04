"""
Example script demonstrating how to use ChromaDB embedding manager.
"""
import asyncio
import os
import shutil
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from wise_nutrition.utils.config import Config
from wise_nutrition.embeddings.chroma_embedding_manager import ChromaEmbeddingManager
from wise_nutrition.embeddings.embedding_factory import get_embedding_manager


def setup_env_for_local_development():
    """Set up environment variables for local development if not already present."""
    if not os.getenv("OPENAI_API_KEY"):
        # Load from .env file
        load_dotenv()
        
        # If still not set, use placeholder for demo
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "sk-your-openai-api-key"
            print("Warning: Using placeholder for OPENAI_API_KEY. Replace with your actual API key.")


def create_sample_documents() -> List[Document]:
    """Create sample nutrition documents for demonstration."""
    return [
        Document(
            page_content="Traditional societies often allowed grains, milk products, and vegetables to ferment via lacto-fermentation. This process preserves food and enhances nutritional value.",
            metadata={
                "section": "Fermentation and Lactic Acid Bacteria",
                "tags": ["fermentation", "gut health", "lactic acid"],
                "source": "Nourishing Traditions, p. 35",
                "chunk_id": "theory_fermentation_lactic"
            }
        ),
        Document(
            page_content="Raw milk contains enzymes and beneficial bacteria that help in the digestion of its nutrients, particularly lactose. Pasteurization destroys these enzymes.",
            metadata={
                "section": "Milk and Dairy Products",
                "tags": ["raw milk", "enzymes", "lactose"],
                "source": "Nourishing Traditions, p. 46",
                "chunk_id": "theory_raw_milk_enzymes"
            }
        ),
        Document(
            page_content="Piima Starter Culture can be used to make a variety of cultured dairy products. It contains specific strains of beneficial bacteria that thrive in milk and create a tangy, probiotic-rich food.",
            metadata={
                "title": "Piima Starter Culture",
                "section": "Cultured Dairy",
                "nutrients": ["Probiotics", "Vitamin A", "Vitamin K2"],
                "chunk_id": "recipe_piima_starter_culture"
            }
        )
    ]


async def run_async_example():
    """Run an example using async methods."""
    print("\nRunning async example...")
    
    # Use a different persist directory for async to avoid conflicts
    persist_directory = os.path.join(os.getcwd(), "chroma_db_async")
    
    # Make sure the directory is writable
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create ChromaDB embedding manager
    manager = ChromaEmbeddingManager(
        config=Config(),
        collection_name="nutrition_collection_async",
        persist_directory=persist_directory
    )
    
    # Create a collection
    await manager.create_collection()
    print("Created collection")
    
    # Add sample documents
    documents = create_sample_documents()
    await manager.add_documents(documents)
    print(f"Added {len(documents)} documents to collection")
    
    # Search for similar documents
    query = "What are the benefits of fermentation for gut health?"
    results = await manager.search_similar(query, k=2)
    
    print(f"\nResults for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")


def run_sync_example():
    """Run an example using synchronous methods."""
    print("\nRunning synchronous example...")
    
    # Use a separate persist directory for sync example
    persist_directory = os.path.join(os.getcwd(), "chroma_db_sync")
    
    # Make sure the directory is writable
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create ChromaDB embedding manager
    manager = ChromaEmbeddingManager(
        config=Config(),
        collection_name="nutrition_collection_sync",
        persist_directory=persist_directory
    )
    
    # Create a collection
    manager.create_collection_sync()
    print("Created collection")
    
    # Add sample documents
    documents = create_sample_documents()
    manager.add_documents_sync(documents)
    print(f"Added {len(documents)} documents to collection")
    
    # Search for similar documents
    query = "Tell me about raw milk and its benefits"
    results = manager.search_similar_sync(query, k=2)
    
    print(f"\nResults for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")


def main():
    """Main function to run the example."""
    # Set up environment for local development
    setup_env_for_local_development()
    
    # Run synchronous example
    run_sync_example()
    
    # Run async example
    asyncio.run(run_async_example())


if __name__ == "__main__":
    main() 