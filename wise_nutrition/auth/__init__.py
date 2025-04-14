"""
Authentication module for Wise Nutrition.
"""

from wise_nutrition.auth.firebase_auth import (
    firebase_auth_manager,
    get_current_user,
    get_current_active_user,
    get_firebase_auth_manager
)

__all__ = [
    "firebase_auth_manager",
    "get_current_user",
    "get_current_active_user",
    "get_firebase_auth_manager"
] 