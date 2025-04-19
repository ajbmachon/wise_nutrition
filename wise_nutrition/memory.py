"""
Memory management for conversation sessions.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
import os
import json

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.base import BaseCheckpointSaver


class FileSystemCheckpointSaver(BaseCheckpointSaver):
    """File system based checkpoint saver for persistent storage."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize with checkpoint directory."""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """Get the file path for a given key."""
        return os.path.join(self.checkpoint_dir, f"{key}.json")
    
    def save(self, key: str, state: Dict[str, Any]) -> None:
        """Save state to file system."""
        file_path = self._get_file_path(key)
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Error saving state to {file_path}: {e}")
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load state from file system."""
        file_path = self._get_file_path(key)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading state from {file_path}: {e}")
        return None
    
    def delete(self, key: str) -> None:
        """Delete state from file system."""
        file_path = self._get_file_path(key)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting state at {file_path}: {e}")


class ConversationState(BaseModel):
    """Schema for conversation thread state."""
    thread_id: str = Field(description="Unique identifier for the conversation thread")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_messages(self) -> List[BaseMessage]:
        """Convert stored messages to LangChain message objects."""
        result = []
        for msg in self.messages:
            msg_type = msg["type"]
            content = msg["content"]
            
            if msg_type == "system":
                result.append(SystemMessage(content=content))
            elif msg_type == "human":
                result.append(HumanMessage(content=content))
            elif msg_type == "ai":
                result.append(AIMessage(content=content))
        return result
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({
            "type": "human",
            "content": content,
            "created_at": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()
    
    def add_ai_message(self, content: str) -> None:
        """Add an AI message to the conversation."""
        self.messages.append({
            "type": "ai",
            "content": content,
            "created_at": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()


class ConversationMemoryManager:
    """
    Manages conversation memory using LangGraph's memory management.
    Handles message filtering and persistence.
    """

    def __init__(
        self,
        max_messages: int = 10,
        checkpoint_dir: str = ".conversation_checkpoints",
        system_message: Optional[str] = None,
        memory_saver: Optional[BaseCheckpointSaver] = None
    ):
        """
        Initialize the conversation memory manager.

        Args:
            max_messages: Maximum number of messages to keep in history (excluding system message)
            checkpoint_dir: Directory to store conversation checkpoints
            system_message: Optional custom system message. If None, uses default.
            memory_saver: Optional pre-configured checkpoint saver. If None, creates FileSystemCheckpointSaver.
        """
        self.max_messages = max_messages
        self.checkpoint_dir = checkpoint_dir
        self._memory_saver = memory_saver or FileSystemCheckpointSaver(checkpoint_dir)
        self._default_system_message = system_message or (
            "I am a nutrition advisor AI. I can help you with questions about "
            "nutrition, vitamins, minerals, and healthy eating habits."
        )
        self._active_sessions: Dict[str, ConversationState] = {}

    def get_memory_saver(self) -> BaseCheckpointSaver:
        """Get the memory saver instance."""
        return self._memory_saver

    def generate_thread_id(self) -> str:
        """Generate a unique thread ID."""
        return str(uuid4())

    def _load_session_state(self, session_id: str) -> ConversationState:
        """
        Load or create session state.
        """
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        try:
            # Try to load from persistent storage
            state_dict = self._memory_saver.load(session_id)
            if state_dict:
                state = ConversationState.model_validate(state_dict)
            else:
                raise ValueError("No state found")
        except Exception as e:
            print(f"Creating new session state for {session_id}: {str(e)}")
            # Create new state if loading fails
            state = ConversationState(thread_id=session_id)
            # Add system message for new conversations
            state.messages.append({
                "type": "system",
                "content": self._default_system_message,
                "created_at": datetime.utcnow().isoformat()
            })

        self._active_sessions[session_id] = state
        return state

    def _save_session_state(self, session_id: str, state: ConversationState):
        """
        Save session state to persistent storage.
        """
        state.updated_at = datetime.utcnow()
        self._active_sessions[session_id] = state
        try:
            self._memory_saver.save(session_id, state.model_dump())
        except Exception as e:
            print(f"Error saving session state: {e}")

    def get_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get chat history for a session, creating it if it doesn't exist.
        Compatible with RunnableWithMessageHistory.
        """
        # Load or create session state
        state = self._load_session_state(session_id)
        
        # Convert state messages to ChatMessageHistory
        history = ChatMessageHistory()
        
        for msg in state.messages:
            msg_type = msg["type"]
            content = msg["content"]
            
            if msg_type == "system":
                history.add_message(SystemMessage(content=content))
            elif msg_type == "human":
                history.add_message(HumanMessage(content=content))
            elif msg_type == "ai":
                history.add_message(AIMessage(content=content))
        
        return history

    def add_message(
        self,
        session_id: str,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the conversation history.
        """
        state = self._load_session_state(session_id)
        
        # Convert message to storable format
        msg_dict = {
            "type": message.__class__.__name__.lower().replace("message", ""),
            "content": message.content,
            "created_at": datetime.utcnow().isoformat()
        }
        if metadata:
            msg_dict["metadata"] = metadata
            
        state.messages.append(msg_dict)
        
        # Apply message filtering
        if len(state.messages) > self.max_messages + 1:  # +1 for system message
            # Keep system message and most recent messages
            system_msg = next((m for m in state.messages if m["type"] == "system"), None)
            recent_msgs = state.messages[-self.max_messages:]
            
            state.messages = ([system_msg] if system_msg else []) + recent_msgs
        
        self._save_session_state(session_id, state)

    def filter_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Filter messages to prevent context window overflow.
        Keeps system messages and most recent messages within max_messages limit.
        """
        if not messages:
            return []

        # Separate system messages and other messages
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        # Keep only the most recent messages (excluding system messages)
        recent_msgs = other_msgs[-self.max_messages:] if other_msgs else []

        # Combine system messages with recent messages
        return system_msgs + recent_msgs

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get metadata for a session.
        """
        state = self._load_session_state(session_id)
        return state.metadata

    def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> None:
        """
        Update session metadata.
        
        Args:
            session_id: The session ID
            metadata: New metadata to set
            merge: If True, merge with existing metadata; if False, replace
        """
        state = self._load_session_state(session_id)
        
        if merge:
            # Deep merge for nested dictionaries
            def deep_merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
                merged = d1.copy()
                for key, value in d2.items():
                    if (
                        key in merged and 
                        isinstance(merged[key], dict) and 
                        isinstance(value, dict)
                    ):
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            
            state.metadata = deep_merge(state.metadata, metadata)
        else:
            state.metadata = metadata
            
        self._save_session_state(session_id, state)

    def clear_session(self, session_id: str) -> None:
        """
        Clear a session's history while maintaining the system message.
        """
        state = self._load_session_state(session_id)
        
        # Keep only the system message if it exists
        system_msg = next((m for m in state.messages if m["type"] == "system"), None)
        state.messages = [system_msg] if system_msg else []
        
        self._save_session_state(session_id, state)
        
    def delete_session(self, session_id: str) -> None:
        """
        Completely delete a session and its history.
        """
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            
        try:
            # Remove from persistent storage
            self._memory_saver.delete(session_id)
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            
    def get_conversation_state(self, session_id: str) -> ConversationState:
        """
        Get the conversation state for a session.
        Creates a new session if it doesn't exist.
        
        Args:
            session_id: The session ID
            
        Returns:
            The conversation state object
        """
        return self._load_session_state(session_id)
        
    def save_conversation_state(self, state: ConversationState) -> None:
        """
        Save the conversation state.
        
        Args:
            state: The conversation state to save
        """
        self._save_session_state(state.thread_id, state)