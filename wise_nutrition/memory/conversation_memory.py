"""
Conversation memory management using LangGraph.
"""
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


class ConversationMemoryManager:
    """
    Manage conversation memory for the nutrition chatbot using LangGraph.
    """
    
    def __init__(
        self,
        memory_saver: Optional[MemorySaver] = None
    ):
        """
        Initialize the conversation memory manager.
        
        Args:
            memory_saver: LangGraph memory saver (if None, a new one will be created)
        """
        # TODO: Check docs and code how to manage memory propperly 
        # TODO: https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/ https://langchain-ai.github.io/langgraph/concepts/memory/ 
        self._memory_saver = memory_saver or MemorySaver()
        
    def get_memory_saver(self) -> MemorySaver:
        """
        Get the memory saver.
        
        Returns:
            LangGraph memory saver instance
        """
        return self._memory_saver
    
    def add_user_message(self, message: str, thread_id: Optional[str] = None) -> str:
        """
        Add a user message to the memory.
        
        Args:
            message: User message string
            thread_id: Optional thread ID for conversation tracking
            
        Returns:
            Thread ID
        """
        pass
    
    def add_ai_message(self, message: str, thread_id: Optional[str] = None) -> str:
        """
        Add an AI message to the memory.
        
        Args:
            message: AI message string
            thread_id: Optional thread ID for conversation tracking
            
        Returns:
            Thread ID
        """
        pass
    
    def get_chat_history(self, thread_id: str) -> List[BaseMessage]:
        """
        Get the chat history for a thread.
        
        Args:
            thread_id: Thread ID
            
        Returns:
            List of chat messages
        """
        pass
    
    def clear(self, thread_id: str) -> None:
        """
        Clear the conversation memory for a thread.
        
        Args:
            thread_id: Thread ID
        """
        pass
    
    def generate_thread_id(self) -> str:
        """
        Generate a new thread ID.
        
        Returns:
            Thread ID string
        """
        return str(uuid4()) 