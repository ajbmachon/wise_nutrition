"""
FastAPI dependencies for creating shared resources.
"""
import os
from typing import Annotated # Use Annotated for Depends

from fastapi import Depends
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.documents import Document

# Import application components
from wise_nutrition.utils.config import Config
from wise_nutrition.rag_chain import NutritionRAGChain
from wise_nutrition.memory import ConversationMemoryManager
from langgraph.checkpoint.memory import MemorySaver # Default saver
# TODO: Add import for persistent saver like SqliteSaver when implemented

# Create a Config instance for use in the dependencies
config = Config()

# --- Dependency Functions --- #

# Singleton pattern for Memory Manager (optional, but often useful)
# Initialize it once here, or use FastAPI's state or lifespan events
# For simplicity, initialize here for now.
# TODO: Consider using SqliteSaver for persistence
_memory_manager_instance = ConversationMemoryManager(memory_saver=MemorySaver())

def get_memory_manager() -> ConversationMemoryManager:
    """Dependency to get the singleton ConversationMemoryManager instance."""
    return _memory_manager_instance

def get_llm() -> Runnable:
    """Dependency to get the language model instance."""
    # TODO: Add error handling for API key
    api_key = config.openai_api_key
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in config/environment.")
        # Optionally raise an HTTPException here if API key is strictly required
        # from fastapi import HTTPException
        # raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    try:
        # Set up LLM with configuration
        return ChatOpenAI(
            model=config.openai_model_default,
            api_key=api_key,
            temperature=0,
            # Add LangSmith tracking metadata
            tags=["nutrition_advisor", "openai"],
            metadata={
                "model": config.openai_model_default,
                "use_case": "nutrition_advisory",
                "service": "wise_nutrition"
            }
        )
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        # Raise or return a fallback? For now, print error.
        # raise HTTPException(status_code=500, detail=f"Could not initialize LLM: {e}")
        # Returning Passthrough as a fallback might hide issues.
        return RunnablePassthrough() # Consider implications of fallback

def get_retriever() -> Runnable:
    """Dependency to get the retriever instance."""
    api_key = config.openai_api_key
    if not api_key:
        print("Warning: OPENAI_API_KEY not found for retriever embeddings.")
        # Consider raising exception if embeddings are critical

    try:
        persist_directory = "chroma_db"
        embedding_function = OpenAIEmbeddings(api_key=api_key)
        
        # Check if the vector store exists
        if not os.path.exists(persist_directory):
            # Create the directory and import data
            os.makedirs(persist_directory)
            
            # Load sample data from files
            documents = []
            
            # Load nutrition.txt
            nutrition_path = "data/samples/nutrition.txt"
            if os.path.exists(nutrition_path):
                with open(nutrition_path, 'r') as f:
                    content = f.read()
                    # Split by double newlines to get sections
                    sections = content.split('\n\n')
                    for section in sections:
                        if section.strip():
                            documents.append(Document(
                                page_content=section.strip(),
                                metadata={"source": "nutrition_sample", "type": "general"}
                            ))
            
            # Load vitamins.json
            vitamins_path = "data/samples/vitamins.json"
            if os.path.exists(vitamins_path):
                import json
                with open(vitamins_path, 'r') as f:
                    vitamins_data = json.load(f)
                    for vitamin in vitamins_data:
                        content = f"Vitamin: {vitamin.get('name', '')}\n"
                        content += f"Description: {vitamin.get('description', '')}\n"
                        content += f"Benefits: {', '.join(vitamin.get('benefits', []))}\n"
                        
                        # Fix: Use food_sources instead of sources
                        food_sources = vitamin.get('food_sources', [])
                        content += f"Food Sources: {', '.join(food_sources)}\n"
                        
                        # Add RDA information
                        rda = vitamin.get('rda', {})
                        if rda:
                            content += "Recommended Daily Allowance:\n"
                            for group, amount in rda.items():
                                content += f"  - {group}: {amount}\n"
                        
                        # Add deficiency symptoms
                        deficiency = vitamin.get('deficiency_symptoms', [])
                        if deficiency:
                            content += f"Deficiency Symptoms: {', '.join(deficiency)}"
                        
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": "vitamins_sample", "type": "vitamin", "name": vitamin.get('name', '')}
                        ))
            
            # Load recipes.json
            recipes_path = "data/samples/recipes.json"
            if os.path.exists(recipes_path):
                import json
                with open(recipes_path, 'r') as f:
                    recipes_data = json.load(f)
                    for recipe in recipes_data:
                        content = f"Recipe: {recipe.get('name', '')}\n"
                        content += f"Description: {recipe.get('description', '')}\n"
                        content += f"Ingredients: {', '.join(recipe.get('ingredients', []))}\n"
                        content += f"Instructions: {recipe.get('instructions', '')}\n"
                        content += f"Nutrition: {recipe.get('nutrition_info', '')}"
                        
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": "recipes_sample", "type": "recipe", "name": recipe.get('name', '')}
                        ))
            
            print(f"Loaded {len(documents)} documents from sample files")
            
            # Create and persist the vector store
            vector_store = Chroma.from_documents(
                documents, embedding_function,
                persist_directory=persist_directory
            )
            vector_store.persist()
            print(f"Vector store created and persisted at {persist_directory}")
        else:
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function
            )
            print(f"Loaded existing vector store from {persist_directory}")
            
        # Return the retriever
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    except Exception as e:
        print(f"Error creating retriever: {e}")
        # Create a small fallback set of documents if vector store creation fails
        fallback_docs = [
            Document(page_content="Vitamin D is essential for calcium absorption and bone health."),
            Document(page_content="Good sources of Vitamin D include sunlight, fatty fish, fortified foods."),
            Document(page_content="Vitamin C supports immune function and is found in citrus fruits."),
            Document(page_content="Iron is important for blood health and can be found in red meat and leafy greens.")
        ]
        fallback_db = Chroma.from_documents(fallback_docs, embedding_function)
        print("Created fallback retriever with basic nutrition information")
        return fallback_db.as_retriever()

def get_rag_chain(
    retriever: Annotated[Runnable, Depends(get_retriever)],
    llm: Annotated[Runnable, Depends(get_llm)],
    memory_manager: Annotated[ConversationMemoryManager, Depends(get_memory_manager)]
) -> NutritionRAGChain:
    """Dependency to create and return the NutritionRAGChain instance."""
    return NutritionRAGChain(
        retriever=retriever,
        llm=llm,
        memory_manager=memory_manager
    ) 