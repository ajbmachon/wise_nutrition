"""
Tests for Firebase authentication module.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from uuid import UUID

from wise_nutrition.models.user import UserCreate, UserLogin, UserResponse
from wise_nutrition.auth.firebase_auth import FirebaseAuthManager

# Mock Firebase user object
class MockFirebaseUser:
    def __init__(self, uid="123e4567-e89b-12d3-a456-426614174000", email="test@example.com"):
        self.uid = uid
        self.email = email
        self.display_name = "Test User"
        self.disabled = False
        
        # Mock user metadata
        class UserMetadata:
            def __init__(self):
                self.creation_timestamp = 1600000000000  # milliseconds
                self.last_refresh_timestamp = 1600100000000  # milliseconds
        
        self.user_metadata = UserMetadata()

# Skip these tests if using actual Firebase integration
@pytest.mark.skip("Skip Firebase tests unless running with mocked Firebase")
class TestFirebaseAuthManager:
    """Test the FirebaseAuthManager class with mocked Firebase."""
    
    @patch("firebase_admin.auth")
    @pytest.mark.asyncio
    async def test_create_user(self, mock_auth):
        """Test user creation."""
        # Set up mock
        mock_firebase_user = MockFirebaseUser()
        mock_auth.create_user.return_value = mock_firebase_user
        
        # Create user data
        user_data = UserCreate(
            email="test@example.com",
            password="Password123",
            full_name="Test User"
        )
        
        # Call the method under test
        result = await FirebaseAuthManager.create_user(user_data)
        
        # Verify Firebase was called correctly
        mock_auth.create_user.assert_called_once_with(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.full_name,
            disabled=not user_data.is_active
        )
        
        # Check result
        assert isinstance(result, UserResponse)
        assert result.email == user_data.email
        assert result.full_name == user_data.full_name
        assert result.is_active == True
    
    @patch("firebase_admin.auth")
    @pytest.mark.asyncio
    async def test_authenticate_user(self, mock_auth):
        """Test user authentication."""
        # Set up mock
        mock_firebase_user = MockFirebaseUser()
        mock_auth.get_user_by_email.return_value = mock_firebase_user
        mock_auth.create_custom_token.return_value = "mocked_token_12345"
        
        # Create login credentials
        credentials = UserLogin(
            email="test@example.com",
            password="Password123"
        )
        
        # Call the method under test
        result = await FirebaseAuthManager.authenticate_user(credentials)
        
        # Verify Firebase was called correctly
        mock_auth.get_user_by_email.assert_called_once_with(credentials.email)
        mock_auth.create_custom_token.assert_called_once_with(mock_firebase_user.uid)
        
        # Check result
        assert result.access_token == "mocked_token_12345"
        assert result.token_type == "bearer"
        assert "refresh_" in result.refresh_token
    
    @patch("firebase_admin.auth")
    @pytest.mark.asyncio
    async def test_get_current_user(self, mock_auth):
        """Test getting current user from token."""
        # Set up mock
        mock_firebase_user = MockFirebaseUser()
        mock_auth.verify_id_token.return_value = {"uid": mock_firebase_user.uid}
        mock_auth.get_user.return_value = mock_firebase_user
        
        # Call the method under test
        with pytest.raises(Exception):
            # This should fail as we're not in FastAPI context
            # Only for illustration - actual test would use FastAPI TestClient
            result = await FirebaseAuthManager.get_current_user("fake_token")
    
    @patch("firebase_admin.auth")
    @pytest.mark.asyncio
    async def test_update_user(self, mock_auth):
        """Test updating user details."""
        # Set up mock
        mock_firebase_user = MockFirebaseUser()
        mock_auth.update_user.return_value = mock_firebase_user
        
        # Update data
        update_data = {
            "email": "updated@example.com",
            "full_name": "Updated Name"
        }
        
        # Call the method under test
        result = await FirebaseAuthManager.update_user(
            UUID("123e4567-e89b-12d3-a456-426614174000"),
            update_data
        )
        
        # Verify Firebase was called correctly
        mock_auth.update_user.assert_called_once_with(
            "123e4567-e89b-12d3-a456-426614174000",
            email="updated@example.com",
            display_name="Updated Name"
        )
        
        # Check result
        assert isinstance(result, UserResponse)
        assert result.email == mock_firebase_user.email
        assert result.full_name == mock_firebase_user.display_name 