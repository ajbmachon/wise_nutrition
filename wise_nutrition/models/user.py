"""
User model and authentication-related schemas.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator
from uuid import uuid4, UUID

class UserBase(BaseModel):
    """Base user model with common fields."""
    email: EmailStr = Field(description="User's email address", examples=["user@example.com"])
    full_name: Optional[str] = Field(None, description="User's full name")
    is_active: bool = Field(True, description="Whether the user is active")
    
class UserCreate(UserBase):
    """Schema for user creation requests."""
    password: str = Field(description="User's password (will be hashed)")
    
    @field_validator('password')
    def validate_password(cls, value):
        """
        Validate password strength.
        At least 8 characters, must contain letters and numbers.
        """
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isalpha() for c in value):
            raise ValueError("Password must contain at least one letter")
        if not any(c.isdigit() for c in value):
            raise ValueError("Password must contain at least one number")
        return value

class UserLogin(BaseModel):
    """Schema for user login requests."""
    email: EmailStr
    password: str

class UserResponse(UserBase):
    """Schema for user responses (without sensitive data)."""
    id: UUID = Field(description="User's unique identifier")
    created_at: datetime = Field(description="When the user was created")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "full_name": "John Doe",
                "is_active": True,
                "created_at": "2023-01-01T00:00:00",
                "last_login": "2023-01-02T12:30:45"
            }]
        }
    }

class UserInDB(UserBase):
    """
    Database representation of a user.
    Includes hashed password and internal fields.
    """
    id: UUID = Field(default_factory=uuid4, description="User's unique identifier")
    hashed_password: str = Field(description="Hashed password")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the user was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the user was last updated")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    email_verified: bool = Field(False, description="Whether the email has been verified")
    verification_token: Optional[str] = Field(None, description="Email verification token")
    reset_token: Optional[str] = Field(None, description="Password reset token")
    reset_token_expires: Optional[datetime] = Field(None, description="When the reset token expires")
    
    model_config = {"arbitrary_types_allowed": True}

class Token(BaseModel):
    """Schema for access token responses."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    
class TokenData(BaseModel):
    """Schema for decoded JWT payload."""
    user_id: UUID
    email: EmailStr
    exp: datetime 