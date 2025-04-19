"""
Tests for the ConversationMemoryManager with LangGraph.
"""
import os
import pytest
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock
from uuid import uuid4, UUID

from wise_nutrition.memory import ConversationMemoryManager, ConversationState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver


class TestConversationMemoryManager:
    """
    Test the ConversationMemoryManager class with LangGraph memory.
    """
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.memory_manager = ConversationMemoryManager(
            checkpoint_dir=self.temp_dir,
            max_messages=3  # Small number for testing
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization with default parameters."""
        manager = ConversationMemoryManager()
        assert isinstance(manager._memory_saver, BaseCheckpointSaver)
        assert manager.max_messages == 10  # Default value
        assert "nutrition advisor AI" in manager._default_system_message
    
    def test_get_memory_saver(self):
        """Test getting the memory saver."""
        memory_saver = self.memory_manager.get_memory_saver()
        assert isinstance(memory_saver, BaseCheckpointSaver)
    
    def test_generate_thread_id(self):
        """Test generating a thread ID."""
        thread_id = self.memory_manager.generate_thread_id()
        assert isinstance(thread_id, str)
        # Should be a valid UUID string
        try:
            UUID(thread_id)  # Try to parse as UUID
            valid_uuid = True
        except ValueError:
            valid_uuid = False
        assert valid_uuid
    
    def test_load_new_session_state(self):
        """Test loading a new session state."""
        session_id = str(uuid4())
        state = self.memory_manager._load_session_state(session_id)
        
        assert isinstance(state, ConversationState)
        assert state.thread_id == session_id
        assert len(state.messages) == 1  # Should have system message
        assert state.messages[0]["type"] == "system"
        assert "nutrition advisor AI" in state.messages[0]["content"]
    
    def test_save_and_load_session_state(self):
        """Test saving and loading session state."""
        session_id = str(uuid4())
        
        # Create and save state
        state = ConversationState(
            thread_id=session_id,
            messages=[{
                "type": "system",
                "content": "Test message",
                "created_at": datetime.utcnow().isoformat()
            }]
        )
        self.memory_manager._save_session_state(session_id, state)
        
        # Load state back
        loaded_state = self.memory_manager._load_session_state(session_id)
        assert loaded_state.thread_id == session_id
        assert len(loaded_state.messages) == 1
        assert loaded_state.messages[0]["type"] == "system"
        assert loaded_state.messages[0]["content"] == "Test message"
    
    def test_get_chat_history(self):
        """Test getting chat history."""
        session_id = str(uuid4())
        
        # Add some messages
        state = self.memory_manager._load_session_state(session_id)
        messages = [
            {"type": "human", "content": "Hello", "created_at": datetime.utcnow().isoformat()},
            {"type": "ai", "content": "Hi there!", "created_at": datetime.utcnow().isoformat()}
        ]
        state.messages.extend(messages)
        self.memory_manager._save_session_state(session_id, state)
        
        # Get history
        history = self.memory_manager.get_chat_history(session_id)
        assert len(history.messages) == 3  # System + 2 messages
        assert isinstance(history.messages[0], SystemMessage)
        assert isinstance(history.messages[1], HumanMessage)
        assert isinstance(history.messages[2], AIMessage)
    
    def test_add_message(self):
        """Test adding messages to history."""
        session_id = str(uuid4())
        
        # Add messages
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")
        
        self.memory_manager.add_message(session_id, human_msg)
        self.memory_manager.add_message(session_id, ai_msg)
        
        # Verify
        history = self.memory_manager.get_chat_history(session_id)
        assert len(history.messages) == 3  # System + 2 messages
        assert history.messages[1].content == "Hello"
        assert history.messages[2].content == "Hi there!"
    
    def test_filter_messages(self):
        """Test message filtering."""
        # Create test messages
        system_msg = SystemMessage(content="System message")
        messages = [
            system_msg,
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Message 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Message 3"),
            AIMessage(content="Response 3")
        ]
        
        # Filter with max_messages=3
        filtered = self.memory_manager.filter_messages(messages)
        
        # Should keep system message and last 3 message pairs
        assert len(filtered) == 4  # System + last 3 messages
        assert filtered[0] == system_msg  # System message preserved
        assert filtered[-3:] == messages[-3:]  # Last 3 messages preserved
    
    def test_session_metadata(self):
        """Test session metadata operations."""
        session_id = str(uuid4())
        
        # Test updating metadata
        metadata = {"user_id": "123", "preferences": {"theme": "dark"}}
        self.memory_manager.update_session_metadata(session_id, metadata)
        
        # Test retrieving metadata
        retrieved = self.memory_manager.get_session_metadata(session_id)
        assert retrieved["user_id"] == "123"
        assert retrieved["preferences"]["theme"] == "dark"
        
        # Test merging metadata
        new_metadata = {"preferences": {"language": "en"}, "new_field": "value"}
        self.memory_manager.update_session_metadata(session_id, new_metadata, merge=True)
        
        merged = self.memory_manager.get_session_metadata(session_id)
        assert merged["preferences"]["theme"] == "dark"  # Original value preserved
        assert merged["preferences"]["language"] == "en"  # New value added
        assert merged["new_field"] == "value"  # New field added
    
    def test_clear_session(self):
        """Test clearing a session."""
        session_id = str(uuid4())
        
        # Add some messages
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")
        self.memory_manager.add_message(session_id, human_msg)
        self.memory_manager.add_message(session_id, ai_msg)
        
        # Clear session
        self.memory_manager.clear_session(session_id)
        
        # Verify only system message remains
        history = self.memory_manager.get_chat_history(session_id)
        assert len(history.messages) == 1
        assert isinstance(history.messages[0], SystemMessage)
    
    def test_delete_session(self):
        """Test deleting a session."""
        session_id = str(uuid4())
        
        # Add some messages
        human_msg = HumanMessage(content="Hello")
        self.memory_manager.add_message(session_id, human_msg)
        
        # Delete session
        self.memory_manager.delete_session(session_id)
        
        # Verify new session is created with fresh state
        history = self.memory_manager.get_chat_history(session_id)
        assert len(history.messages) == 1  # Only system message
        assert isinstance(history.messages[0], SystemMessage)

    @pytest.mark.asyncio
    async def test_add_user_message(self):
        """Test adding a user message."""
        # Test add_user_message here
        thread_id = "test-thread-id"
        message = "Hello, nutrition advisor!"
        # Test with provided thread_id and test return value
    
    @pytest.mark.asyncio
    async def test_add_ai_message(self):
        """Test adding an AI message."""
        # Test add_ai_message here
        thread_id = "test-thread-id"
        message = "Here is nutrition advice."
        # Test with provided thread_id and test return value
    
    @pytest.mark.asyncio
    async def test_get_chat_history(self):
        """Test getting chat history."""
        # Test get_chat_history here
        thread_id = "test-thread-id"
        # Setup mock messages in the memory and test retrieval
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing conversation memory."""
        # Test clear here
        thread_id = "test-thread-id"
        # Test clearing memory for a specific thread_id 

def test_generate_thread_id():
    """Test that thread ID generation produces valid UUIDs."""
    manager = ConversationMemoryManager()
    thread_id = manager.generate_thread_id()
    
    # Verify it's a valid UUID string
    try:
        uuid_obj = UUID(thread_id)
        assert str(uuid_obj) == thread_id
    except ValueError:
        pytest.fail("Generated thread ID is not a valid UUID")


def test_new_session_initialization():
    """Test that new sessions are initialized with system message."""
    manager = ConversationMemoryManager()
    session_id = manager.generate_thread_id()
    
    history = manager.get_chat_history(session_id)
    messages = history.messages
    
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "nutrition advisor" in messages[0].content.lower()


def test_custom_system_message():
    """Test initialization with custom system message."""
    custom_msg = "Custom system message for testing"
    manager = ConversationMemoryManager(system_message=custom_msg)
    session_id = manager.generate_thread_id()
    
    history = manager.get_chat_history(session_id)
    assert history.messages[0].content == custom_msg


def test_add_and_retrieve_messages():
    """Test adding and retrieving messages."""
    manager = ConversationMemoryManager()
    session_id = manager.generate_thread_id()
    
    # Add messages
    human_msg = HumanMessage(content="What foods are high in vitamin C?")
    ai_msg = AIMessage(content="Citrus fruits, strawberries, and bell peppers are high in vitamin C.")
    
    manager.add_message(session_id, human_msg)
    manager.add_message(session_id, ai_msg)
    
    # Retrieve and verify
    history = manager.get_chat_history(session_id)
    messages = history.messages
    
    assert len(messages) == 3  # System + Human + AI
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert messages[1].content == human_msg.content
    assert messages[2].content == ai_msg.content


def test_message_filtering():
    """Test that message filtering works correctly."""
    manager = ConversationMemoryManager(max_messages=2)
    session_id = manager.generate_thread_id()
    
    # Add more messages than the limit
    messages = [
        HumanMessage(content=f"Human message {i}")
        for i in range(4)
    ]
    for msg in messages:
        manager.add_message(session_id, msg)
    
    history = manager.get_chat_history(session_id)
    all_messages = history.messages
    
    # Should have system message + last 2 human messages
    assert len(all_messages) == 3
    assert isinstance(all_messages[0], SystemMessage)
    assert all_messages[1].content == "Human message 2"
    assert all_messages[2].content == "Human message 3"


def test_filter_messages_method():
    """Test the filter_messages method directly."""
    manager = ConversationMemoryManager(max_messages=2)
    
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="First message"),
        AIMessage(content="Second message"),
        HumanMessage(content="Third message")
    ]
    
    filtered = manager.filter_messages(messages)
    
    # Should keep system message + last 2 messages
    assert len(filtered) == 3
    assert isinstance(filtered[0], SystemMessage)
    assert filtered[1].content == "Second message"
    assert filtered[2].content == "Third message"


def test_session_metadata():
    """Test session metadata operations."""
    manager = ConversationMemoryManager()
    session_id = manager.generate_thread_id()
    
    # Test initial empty metadata
    initial_metadata = manager.get_session_metadata(session_id)
    assert initial_metadata == {}
    
    # Test updating metadata
    preferences = {"preferences": {"theme": "dark", "notifications": True}}
    manager.update_session_metadata(session_id, preferences)
    
    metadata = manager.get_session_metadata(session_id)
    assert metadata["preferences"]["theme"] == "dark"
    assert metadata["preferences"]["notifications"] is True
    
    # Test merging metadata
    new_prefs = {"preferences": {"language": "en", "theme": "light"}}
    manager.update_session_metadata(session_id, new_prefs, merge=True)
    
    merged = manager.get_session_metadata(session_id)
    assert merged["preferences"]["theme"] == "light"  # Updated
    assert merged["preferences"]["notifications"] is True  # Preserved
    assert merged["preferences"]["language"] == "en"  # Added
    
    # Test replacing metadata
    new_metadata = {"new_key": "new_value"}
    manager.update_session_metadata(session_id, new_metadata, merge=False)
    
    replaced = manager.get_session_metadata(session_id)
    assert replaced == new_metadata
    assert "preferences" not in replaced


def test_clear_session():
    """Test clearing a session while preserving system message."""
    manager = ConversationMemoryManager()
    session_id = manager.generate_thread_id()
    
    # Add some messages
    manager.add_message(session_id, HumanMessage(content="Test message"))
    manager.add_message(session_id, AIMessage(content="Test response"))
    
    # Clear the session
    manager.clear_session(session_id)
    
    # Verify only system message remains
    history = manager.get_chat_history(session_id)
    assert len(history.messages) == 1
    assert isinstance(history.messages[0], SystemMessage)


def test_delete_session():
    """Test completely deleting a session."""
    manager = ConversationMemoryManager()
    session_id = manager.generate_thread_id()
    
    # Add a message and metadata
    manager.add_message(session_id, HumanMessage(content="Test message"))
    manager.update_session_metadata(session_id, {"test": "data"})
    
    # Delete the session
    manager.delete_session(session_id)
    
    # Verify new session is created with fresh state
    history = manager.get_chat_history(session_id)
    metadata = manager.get_session_metadata(session_id)
    
    assert len(history.messages) == 1  # Only system message
    assert isinstance(history.messages[0], SystemMessage)
    assert metadata == {} 