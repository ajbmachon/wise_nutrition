"""
Conversation memory management.
"""
from typing import Dict, List, Any, Optional

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage


class ConversationMemoryManager:
    """
    Manage conversation memory for the nutrition chatbot.
    """
    
    def __init__(
        self,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: str = "answer"
    ):
        """
        Initialize the conversation memory manager.
        
        Args:
            memory_key: Key to store the memory under
            return_messages: Whether to return messages or strings
            output_key: Key for storing outputs
        """
        pass
    
    def get_memory(self) -> ConversationBufferMemory:
        """
        Get the conversation memory.
        
        Returns:
            Conversation memory instance
        """
        pass
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the memory.
        
        Args:
            message: User message string
        """
        pass
    
    def add_ai_message(self, message: str) -> None:
        """
        Add an AI message to the memory.
        
        Args:
            message: AI message string
        """
        pass
    
    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get the chat history.
        
        Returns:
            List of chat messages
        """
        pass
    
    def clear(self) -> None:
        """
        Clear the conversation memory.
        """
        pass 