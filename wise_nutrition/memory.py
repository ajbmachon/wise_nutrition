"""
Conversation memory management using LangGraph state and checkpoints.
"""
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# --- Define Conversation State Schema --- #

class ConversationState(TypedDict):
    """
    Represents the state of a conversation thread.
    Uses the format expected by RunnableWithMessageHistory.
    """
    # The list of messages in the conversation history
    messages: List[BaseMessage]
    # Potentially add other state variables here if needed later
    # e.g., user_profile: Dict[str, Any]
    # e.g., current_summary: str

# --- Memory Manager Class --- #

class ConversationMemoryManager:
    """
    Provides access to the LangGraph memory saver (checkpointer).
    Manages the persistence layer for conversation state.
    Note: Direct history manipulation might be handled by RunnableWithMessageHistory in the chain.
    This class primarily provides the configured checkpointer.

    Persistence Note (Subtask 4.4):
    The default `MemorySaver` is in-memory only and will lose history when the app restarts.
    For actual persistence, initialize this class with a persistent checkpointer like
    `SqliteSaver.from_conn_string(":memory:")` (for in-memory sqlite) or a file path
    `SqliteSaver.from_conn_string("checkpoints.sqlite")`, or other persistent backends
    (e.g., RedisSaver, PostgresSaver).
    """

    def __init__(
        self,
        memory_saver: Optional[MemorySaver] = None
    ):
        """
        Initialize the conversation memory manager.

        Args:
            memory_saver: LangGraph memory saver instance.
                          If None, a default MemorySaver (in-memory) will be created.
        """
        # TODO: Check docs and code how to manage memory propperly
        # TODO: https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/ https://langchain-ai.github.io/langgraph/concepts/memory/
        # Using MemorySaver as the default in-memory backend.
        # For persistence, consider other backends like SqliteSaver, RedisSaver, etc.
        # See Persistence Note in class docstring.
        self._memory_saver = memory_saver or MemorySaver()

    def get_memory_saver(self) -> MemorySaver:
        """
        Get the configured memory saver instance (checkpointer).

        Returns:
            The LangGraph memory saver instance.
        """
        return self._memory_saver

    # --- Utility methods (REMOVED - Prefer direct use of RunnableWithMessageHistory) --- #

    # def add_user_message(self, message: str, thread_id: Optional[str] = None) -> str:
    #     print("Warning: add_user_message called directly. Prefer chain's memory handling.")
    #     return thread_id or self.generate_thread_id()
    #
    # def add_ai_message(self, message: str, thread_id: Optional[str] = None) -> str:
    #     print("Warning: add_ai_message called directly. Prefer chain's memory handling.")
    #     return thread_id or self.generate_thread_id()
    #
    # def get_chat_history(self, thread_id: str) -> List[BaseMessage]:
    #     print(f"Warning: get_chat_history called directly. Getting state for {thread_id}")
    #     config = {"configurable": {"thread_id": thread_id}}
    #     # This assumes the checkpointer stores state compatible with ConversationState
    #     state = self._memory_saver.get(config)
    #     return state.get("messages", []) if state else []
    #
    # def clear(self, thread_id: str) -> None:
    #     print(f"Warning: clear called. Actual clearing depends on checkpointer backend. Thread: {thread_id}")
    #     pass

    def generate_thread_id(self) -> str:
        """
        Generate a new unique thread ID.

        Returns:
            A UUID string representing the thread ID.
        """
        return str(uuid4()) 