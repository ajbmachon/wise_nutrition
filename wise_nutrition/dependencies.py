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
        return ChatOpenAI(
            model=config.openai_model_default,
            api_key=api_key,
            temperature=0
        )
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        # Raise or return a fallback? For now, print error.
        # raise HTTPException(status_code=500, detail=f"Could not initialize LLM: {e}")
        # Returning Passthrough as a fallback might hide issues.
        return RunnablePassthrough() # Consider implications of fallback

def get_retriever() -> Runnable:
    """Dependency to get the retriever instance."""
    # TODO: Replace with actual retriever implementation (Task 2 refinement)
    # Using the same dummy logic as before for now
    api_key = config.openai_api_key
    if not api_key:
        print("Warning: OPENAI_API_KEY not found for retriever embeddings.")
        # Consider raising exception if embeddings are critical

    try:
        persist_directory = "chroma_db_dummy"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            dummy_docs = [Document(page_content="dummy content")]
            # TODO: Add error handling for API key
            embedding_function = OpenAIEmbeddings(api_key=api_key)
            dummy_db = Chroma.from_documents(
                dummy_docs, embedding_function,
                persist_directory=persist_directory
            )
            dummy_db.persist()
        else:
            embedding_function = OpenAIEmbeddings(api_key=api_key)
            dummy_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function
            )
        return dummy_db.as_retriever()
    except Exception as e:
        print(f"Error creating dummy retriever dependency: {e}. Using Passthrough.")
        return RunnablePassthrough()

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