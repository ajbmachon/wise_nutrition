"""
Firebase authentication and user management module.
"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from wise_nutrition.models.user import UserCreate, UserResponse, UserInDB, Token, TokenData, UserLogin

# Check if we're in development mode
DEV_MODE = os.environ.get("WISE_NUTRITION_DEV_MODE", "true").lower() == "true"

# Initialize Firebase with credentials if not in dev mode
if not DEV_MODE:
    cred_path = os.path.join(os.getcwd(), "firebase_credentials.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_app = firebase_admin.initialize_app(cred)
    else:
        raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")
else:
    print("⚠️ Running in DEVELOPMENT MODE - using mock authentication ⚠️")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

class FirebaseAuthManager:
    """Handles Firebase authentication and user management."""
    
    @staticmethod
    async def create_user(user_data: UserCreate) -> UserResponse:
        """
        Create a new user in Firebase.
        
        Args:
            user_data: User creation data
            
        Returns:
            User response data
            
        Raises:
            HTTPException: If user creation fails
        """
        try:
            # Create the user in Firebase Authentication
            firebase_user = auth.create_user(
                email=user_data.email,
                password=user_data.password,
                display_name=user_data.full_name or "",
                disabled=not user_data.is_active
            )
            
            # Convert to our application's user model
            user_response = UserResponse(
                id=UUID(firebase_user.uid),
                email=firebase_user.email,
                full_name=firebase_user.display_name,
                is_active=not firebase_user.disabled,
                created_at=datetime.utcnow(),
                last_login=None
            )
            
            return user_response
            
        except firebase_admin.exceptions.FirebaseError as e:
            # Handle Firebase-specific errors
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            elif "WEAK_PASSWORD" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password is too weak"
                )
            else:
                # Log the full error for debugging
                print(f"Firebase error: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user"
                )
    
    @staticmethod
    async def authenticate_user(user_credentials: UserLogin) -> Token:
        """
        Authenticate a user with Firebase and return a token.
        
        Args:
            user_credentials: User login credentials
            
        Returns:
            Authentication token
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # We can't directly authenticate with Firebase Admin SDK
            # In a real application, you would use Firebase Auth REST API
            # For this example, we'll verify the user exists and return a custom token
            user = auth.get_user_by_email(user_credentials.email)
            
            # In production, you'd use Firebase Auth REST API to verify password
            # This is a simplification for demo purposes
            custom_token = auth.create_custom_token(user.uid)
            
            # Convert bytes to string if needed
            if isinstance(custom_token, bytes):
                custom_token = custom_token.decode('utf-8')
            
            # Create a refresh token (in production you'd use Firebase Auth REST API)
            refresh_token = f"refresh_{custom_token[-10:]}"
            
            return Token(
                access_token=custom_token,
                refresh_token=refresh_token,
                token_type="bearer"
            )
            
        except firebase_admin.exceptions.FirebaseError as e:
            # Handle Firebase-specific errors
            error_message = str(e)
            if "USER_NOT_FOUND" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                # Log the full error for debugging
                print(f"Firebase error during authentication: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication failed"
                )
    
    @staticmethod
    async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
        """
        Get the current authenticated user from a token.
        
        Args:
            token: JWT token
            
        Returns:
            Current user data
            
        Raises:
            HTTPException: If token is invalid
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            # Verify Firebase token
            decoded_token = auth.verify_id_token(token)
            
            # Get user details
            firebase_user = auth.get_user(decoded_token["uid"])
            
            # Convert to our application's user model
            user = UserResponse(
                id=UUID(firebase_user.uid),
                email=firebase_user.email,
                full_name=firebase_user.display_name,
                is_active=not firebase_user.disabled,
                created_at=datetime.utcfromtimestamp(firebase_user.user_metadata.creation_timestamp / 1000),
                last_login=datetime.utcfromtimestamp(firebase_user.user_metadata.last_refresh_timestamp / 1000) if firebase_user.user_metadata.last_refresh_timestamp else None
            )
            
            return user
            
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            raise credentials_exception
    
    @staticmethod
    async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        """
        Get the current active user.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            Current active user
            
        Raises:
            HTTPException: If user is inactive
        """
        if not current_user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    @staticmethod
    async def update_user(user_id: UUID, update_data: Dict[str, Any]) -> UserResponse:
        """
        Update a user in Firebase.
        
        Args:
            user_id: User ID
            update_data: Data to update
            
        Returns:
            Updated user data
            
        Raises:
            HTTPException: If update fails
        """
        try:
            # Extract updatable fields
            update_args = {}
            
            if "email" in update_data:
                update_args["email"] = update_data["email"]
            
            if "full_name" in update_data:
                update_args["display_name"] = update_data["full_name"]
            
            if "is_active" in update_data:
                update_args["disabled"] = not update_data["is_active"]
            
            if "password" in update_data:
                update_args["password"] = update_data["password"]
            
            # Update user in Firebase
            firebase_user = auth.update_user(str(user_id), **update_args)
            
            # Convert to our application's user model
            user_response = UserResponse(
                id=UUID(firebase_user.uid),
                email=firebase_user.email,
                full_name=firebase_user.display_name,
                is_active=not firebase_user.disabled,
                created_at=datetime.utcfromtimestamp(firebase_user.user_metadata.creation_timestamp / 1000),
                last_login=datetime.utcfromtimestamp(firebase_user.user_metadata.last_refresh_timestamp / 1000) if firebase_user.user_metadata.last_refresh_timestamp else None
            )
            
            return user_response
            
        except firebase_admin.exceptions.FirebaseError as e:
            # Handle Firebase-specific errors
            error_message = str(e)
            if "USER_NOT_FOUND" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {user_id} not found"
                )
            else:
                # Log the full error for debugging
                print(f"Firebase error during user update: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update user"
                )
    
    @staticmethod
    async def delete_user(user_id: UUID) -> None:
        """
        Delete a user from Firebase.
        
        Args:
            user_id: User ID
            
        Raises:
            HTTPException: If deletion fails
        """
        try:
            auth.delete_user(str(user_id))
        except firebase_admin.exceptions.FirebaseError as e:
            # Handle Firebase-specific errors
            error_message = str(e)
            if "USER_NOT_FOUND" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {user_id} not found"
                )
            else:
                # Log the full error for debugging
                print(f"Firebase error during user deletion: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to delete user"
                )
    
    @staticmethod
    async def send_password_reset_email(email: str) -> None:
        """
        Send a password reset email.
        
        Args:
            email: User email
            
        Raises:
            HTTPException: If operation fails
        """
        try:
            # In Firebase Admin SDK, we can't directly send password reset emails
            # In a real application, you would use Firebase Auth REST API
            # For this example, we'll just verify the user exists
            auth.get_user_by_email(email)
            
            # In production, you'd call Firebase Auth REST API here
            # This is a placeholder
            print(f"Password reset email would be sent to {email}")
            
        except firebase_admin.exceptions.FirebaseError as e:
            # Handle Firebase-specific errors
            error_message = str(e)
            if "USER_NOT_FOUND" in error_message:
                # For security reasons, don't reveal if the email exists
                pass
            else:
                # Log the full error for debugging
                print(f"Firebase error when sending reset email: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to process password reset request"
                )
    
    @staticmethod
    async def verify_email(email: str, code: str) -> bool:
        """
        Verify a user's email.
        
        Args:
            email: User email
            code: Verification code
            
        Returns:
            Whether verification was successful
            
        Raises:
            HTTPException: If verification fails
        """
        try:
            # In Firebase Admin SDK, we can't directly verify emails
            # In a real application, you would use Firebase Auth REST API
            # For this example, we'll just verify the user exists
            user = auth.get_user_by_email(email)
            
            # In production, you'd call Firebase Auth REST API here
            # This is a placeholder - always returns True for demo
            return True
            
        except firebase_admin.exceptions.FirebaseError as e:
            # Handle Firebase-specific errors
            error_message = str(e)
            if "USER_NOT_FOUND" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            else:
                # Log the full error for debugging
                print(f"Firebase error during email verification: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to verify email"
                )

# Create singleton instance
firebase_auth_manager = FirebaseAuthManager()

# Dependency for routes
def get_firebase_auth_manager():
    """Dependency to get the Firebase Auth Manager."""
    return firebase_auth_manager

# Create a mock user for development
async def get_mock_user() -> UserResponse:
    """Return a mock user for development purposes."""
    return UserResponse(
        id=uuid4(),
        email="dev@example.com",
        full_name="Development User",
        is_active=True,
        created_at=datetime.now(),
        last_login=datetime.now()
    )

# Auth dependencies - use mock in dev mode
if DEV_MODE:
    # In development mode, always return the mock user
    get_current_user = get_mock_user
    get_current_active_user = get_mock_user
else:
    # In production, use the real Firebase authentication
    get_current_user = FirebaseAuthManager.get_current_user
    get_current_active_user = FirebaseAuthManager.get_current_active_user