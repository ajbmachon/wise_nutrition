"""
Authentication router for user registration, login, and management.
"""
from typing import Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from wise_nutrition.models.user import UserCreate, UserResponse, UserLogin, Token
from wise_nutrition.auth.firebase_auth import (
    firebase_auth_manager,
    get_current_user,
    get_current_active_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Register a new user.
    """
    return await firebase_auth_manager.create_user(user_data)

@router.post("/login", response_model=Token)
async def login_user(user_credentials: UserLogin):
    """
    Authenticate user and return token.
    
    """
    return await firebase_auth_manager.authenticate_user(user_credentials)

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_active_user)):
    """
    Get current authenticated user information.
    """
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_user_me(
    update_data: Dict[str, Any],
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Update current user information.
    Allows updating email, full_name, and password.
    """
    return await firebase_auth_manager.update_user(current_user.id, update_data)

@router.post("/password-reset")
async def request_password_reset(email_data: Dict[str, str]):
    """
    Send password reset email.
    """
    if "email" not in email_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required"
        )
    await firebase_auth_manager.send_password_reset_email(email_data["email"])
    return {"message": "If your email is registered, you will receive a password reset link."}

@router.post("/verify-email", response_model=Dict[str, bool])
async def verify_email(verification_data: Dict[str, str]):
    """
    Verify user email with verification code.
    """
    if "email" not in verification_data or "code" not in verification_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and verification code are required"
        )
    
    is_verified = await firebase_auth_manager.verify_email(
        verification_data["email"],
        verification_data["code"]
    )
    
    return {"verified": is_verified}

# Optional: Admin-only endpoints
@router.get("/users/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: UUID,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get user information by ID (admin only).
    """
    # TODO: Add admin check
    try:
        # This function is not implemented in firebase_auth yet
        # In a real application, you would call Firebase Admin SDK
        # For this example, we'll just return a message
        return {"message": f"Admin: would get user with ID {user_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}"
        )

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Delete a user (admin only).
    """
    # TODO: Add admin check
    await firebase_auth_manager.delete_user(user_id)
    return {"message": f"User with ID {user_id} deleted"} 