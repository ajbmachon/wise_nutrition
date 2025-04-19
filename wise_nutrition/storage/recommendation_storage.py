"""
Storage service for recommendations, tags, and categories.
"""
import json
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from uuid import UUID, uuid4

import firebase_admin
from firebase_admin import firestore, credentials
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1.collection import CollectionReference

from wise_nutrition.models.recommendation import (
    RecommendationInDB,
    TagInDB,
    CategoryInDB,
    RecommendationCreate,
    RecommendationUpdate,
    Recommendation,
    Tag,
    Category,
    TagCreate,
    CategoryCreate
)
from wise_nutrition.models.user import UserResponse


class RecommendationStorageService:
    """Service for handling recommendation storage operations in Firestore."""
    
    def __init__(self, db: Optional[firestore.Client] = None):
        """
        Initialize the storage service.
        
        Args:
            db: Firestore client instance
        """
        self.db = db or self._get_firestore_client()
        self.recommendations_collection = "recommendations"
        self.tags_collection = "tags"
        self.categories_collection = "categories"
        
    def _get_firestore_client(self) -> firestore.Client:
        """
        Get or initialize Firebase client.
        """
        try:
            # Use existing app if available
            return firestore.client()
        except ValueError:
            # Initialize app if not already done
            cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
            
            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")
                
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            return firestore.client()
    
    # --- Utility Methods --- #
    
    def _convert_to_dict(self, obj: Union[RecommendationInDB, TagInDB, CategoryInDB]) -> Dict[str, Any]:
        """
        Convert an object to a Firestore-compatible dict.
        
        Args:
            obj: The object to convert
            
        Returns:
            Dict ready for Firestore storage
        """
        # Convert model to dict
        data = obj.model_dump()
        
        # Handle UUID serialization
        for key, value in data.items():
            if isinstance(value, UUID):
                data[key] = str(value)
            elif isinstance(value, list) and value and isinstance(value[0], UUID):
                data[key] = [str(v) for v in value]
            elif isinstance(value, datetime):
                # Firestore handles datetime objects natively
                pass
                
        return data
    
    def _recommendation_from_doc(self, doc: Dict[str, Any]) -> RecommendationInDB:
        """Convert a Firestore document to a RecommendationInDB object."""
        # Convert string UUIDs back to UUID objects
        if "id" in doc:
            doc["id"] = UUID(doc["id"])
        if "user_id" in doc:
            doc["user_id"] = UUID(doc["user_id"])
        if "category_id" in doc and doc["category_id"]:
            doc["category_id"] = UUID(doc["category_id"])
        if "tag_ids" in doc:
            doc["tag_ids"] = [UUID(tag_id) for tag_id in doc["tag_ids"]]
            
        return RecommendationInDB.model_validate(doc)
    
    def _tag_from_doc(self, doc: Dict[str, Any]) -> TagInDB:
        """Convert a Firestore document to a TagInDB object."""
        if "id" in doc:
            doc["id"] = UUID(doc["id"])
        return TagInDB.model_validate(doc)
    
    def _category_from_doc(self, doc: Dict[str, Any]) -> CategoryInDB:
        """Convert a Firestore document to a CategoryInDB object."""
        if "id" in doc:
            doc["id"] = UUID(doc["id"])
        return CategoryInDB.model_validate(doc)
    
    # --- Tag Operations --- #
    
    async def create_tag(self, tag_data: TagCreate) -> Tag:
        """
        Create a new tag.
        
        Args:
            tag_data: Tag creation data
            
        Returns:
            Created tag
        """
        # Create tag model
        tag_id = uuid4()
        tag = TagInDB(
            id=tag_id,
            name=tag_data.name,
            description=tag_data.description,
            created_at=datetime.utcnow()
        )
        
        # Save to Firestore
        tag_ref = self.db.collection(self.tags_collection).document(str(tag_id))
        tag_ref.set(self._convert_to_dict(tag))
        
        # Return response model
        return Tag(
            id=tag.id,
            name=tag.name,
            description=tag.description,
            created_at=tag.created_at
        )
    
    async def get_tag(self, tag_id: UUID) -> Optional[Tag]:
        """
        Get a tag by ID.
        
        Args:
            tag_id: Tag ID
            
        Returns:
            Tag if found, None otherwise
        """
        tag_ref = self.db.collection(self.tags_collection).document(str(tag_id))
        tag_doc = tag_ref.get()
        
        if not tag_doc.exists:
            return None
            
        tag = self._tag_from_doc(tag_doc.to_dict())
        
        return Tag(
            id=tag.id,
            name=tag.name,
            description=tag.description,
            created_at=tag.created_at
        )
    
    async def get_tags(self) -> List[Tag]:
        """
        Get all tags.
        
        Returns:
            List of tags
        """
        tags_ref = self.db.collection(self.tags_collection)
        tags = []
        
        for tag_doc in tags_ref.stream():
            tag = self._tag_from_doc(tag_doc.to_dict())
            tags.append(Tag(
                id=tag.id,
                name=tag.name,
                description=tag.description,
                created_at=tag.created_at
            ))
            
        return tags
    
    async def update_tag(self, tag_id: UUID, tag_data: TagCreate) -> Optional[Tag]:
        """
        Update a tag.
        
        Args:
            tag_id: Tag ID
            tag_data: Updated tag data
            
        Returns:
            Updated tag if successful, None otherwise
        """
        tag_ref = self.db.collection(self.tags_collection).document(str(tag_id))
        tag_doc = tag_ref.get()
        
        if not tag_doc.exists:
            return None
            
        # Get existing tag
        existing_tag = self._tag_from_doc(tag_doc.to_dict())
        
        # Update fields
        existing_tag.name = tag_data.name
        existing_tag.description = tag_data.description
        
        # Save changes
        tag_ref.update(self._convert_to_dict(existing_tag))
        
        # Return updated tag
        return Tag(
            id=existing_tag.id,
            name=existing_tag.name,
            description=existing_tag.description,
            created_at=existing_tag.created_at
        )
    
    async def delete_tag(self, tag_id: UUID) -> bool:
        """
        Delete a tag.
        
        Args:
            tag_id: Tag ID
            
        Returns:
            True if deleted, False otherwise
        """
        tag_ref = self.db.collection(self.tags_collection).document(str(tag_id))
        tag_doc = tag_ref.get()
        
        if not tag_doc.exists:
            return False
            
        # Remove tag from all recommendations that have it
        recommendations_ref = self.db.collection(self.recommendations_collection)
        query = recommendations_ref.where("tag_ids", "array_contains", str(tag_id))
        
        for rec_doc in query.stream():
            rec_ref = recommendations_ref.document(rec_doc.id)
            rec_data = rec_doc.to_dict()
            if "tag_ids" in rec_data:
                rec_data["tag_ids"] = [tid for tid in rec_data["tag_ids"] if tid != str(tag_id)]
                rec_ref.update({"tag_ids": rec_data["tag_ids"]})
                
        # Delete the tag
        tag_ref.delete()
        return True
    
    # --- Category Operations --- #
    
    async def create_category(self, category_data: CategoryCreate) -> Category:
        """
        Create a new category.
        
        Args:
            category_data: Category creation data
            
        Returns:
            Created category
        """
        # Create category model
        category_id = uuid4()
        category = CategoryInDB(
            id=category_id,
            name=category_data.name,
            description=category_data.description,
            created_at=datetime.utcnow()
        )
        
        # Save to Firestore
        category_ref = self.db.collection(self.categories_collection).document(str(category_id))
        category_ref.set(self._convert_to_dict(category))
        
        # Return response model
        return Category(
            id=category.id,
            name=category.name,
            description=category.description,
            created_at=category.created_at
        )
    
    async def get_category(self, category_id: UUID) -> Optional[Category]:
        """
        Get a category by ID.
        
        Args:
            category_id: Category ID
            
        Returns:
            Category if found, None otherwise
        """
        category_ref = self.db.collection(self.categories_collection).document(str(category_id))
        category_doc = category_ref.get()
        
        if not category_doc.exists:
            return None
            
        category = self._category_from_doc(category_doc.to_dict())
        
        return Category(
            id=category.id,
            name=category.name,
            description=category.description,
            created_at=category.created_at
        )
    
    async def get_categories(self) -> List[Category]:
        """
        Get all categories.
        
        Returns:
            List of categories
        """
        categories_ref = self.db.collection(self.categories_collection)
        categories = []
        
        for category_doc in categories_ref.stream():
            category = self._category_from_doc(category_doc.to_dict())
            categories.append(Category(
                id=category.id,
                name=category.name,
                description=category.description,
                created_at=category.created_at
            ))
            
        return categories
    
    async def update_category(self, category_id: UUID, category_data: CategoryCreate) -> Optional[Category]:
        """
        Update a category.
        
        Args:
            category_id: Category ID
            category_data: Updated category data
            
        Returns:
            Updated category if successful, None otherwise
        """
        category_ref = self.db.collection(self.categories_collection).document(str(category_id))
        category_doc = category_ref.get()
        
        if not category_doc.exists:
            return None
            
        # Get existing category
        existing_category = self._category_from_doc(category_doc.to_dict())
        
        # Update fields
        existing_category.name = category_data.name
        existing_category.description = category_data.description
        
        # Save changes
        category_ref.update(self._convert_to_dict(existing_category))
        
        # Return updated category
        return Category(
            id=existing_category.id,
            name=existing_category.name,
            description=existing_category.description,
            created_at=existing_category.created_at
        )
    
    async def delete_category(self, category_id: UUID) -> bool:
        """
        Delete a category.
        
        Args:
            category_id: Category ID
            
        Returns:
            True if deleted, False otherwise
        """
        category_ref = self.db.collection(self.categories_collection).document(str(category_id))
        category_doc = category_ref.get()
        
        if not category_doc.exists:
            return False
            
        # Update all recommendations that use this category
        recommendations_ref = self.db.collection(self.recommendations_collection)
        query = recommendations_ref.where("category_id", "==", str(category_id))
        
        for rec_doc in query.stream():
            rec_ref = recommendations_ref.document(rec_doc.id)
            rec_ref.update({"category_id": None})
                
        # Delete the category
        category_ref.delete()
        return True
    
    # --- Recommendation Operations --- #
    
    async def create_recommendation(self, user_id: UUID, data: RecommendationCreate) -> Recommendation:
        """
        Create a new recommendation.
        
        Args:
            user_id: User ID
            data: Recommendation creation data
            
        Returns:
            Created recommendation
        """
        # Create recommendation model
        rec_id = uuid4()
        now = datetime.utcnow()
        
        recommendation = RecommendationInDB(
            id=rec_id,
            user_id=user_id,
            title=data.title,
            content=data.content,
            sources=data.sources,
            metadata=data.metadata,
            tag_ids=data.tag_ids or [],
            category_id=data.category_id,
            created_at=now,
            updated_at=now
        )
        
        # Save to Firestore
        rec_ref = self.db.collection(self.recommendations_collection).document(str(rec_id))
        rec_ref.set(self._convert_to_dict(recommendation))
        
        # Get associated tags and category for response
        tags = []
        if recommendation.tag_ids:
            for tag_id in recommendation.tag_ids:
                tag = await self.get_tag(tag_id)
                if tag:
                    tags.append(tag)
        
        category = None
        if recommendation.category_id:
            category = await self.get_category(recommendation.category_id)
        
        # Return response model
        return Recommendation(
            id=recommendation.id,
            user_id=recommendation.user_id,
            title=recommendation.title,
            content=recommendation.content,
            sources=recommendation.sources,
            metadata=recommendation.metadata,
            tags=tags,
            category=category,
            created_at=recommendation.created_at,
            updated_at=recommendation.updated_at
        )
    
    async def get_recommendation(self, rec_id: UUID, user_id: Optional[UUID] = None) -> Optional[Recommendation]:
        """
        Get a recommendation by ID, optionally filtering by user_id.
        
        Args:
            rec_id: Recommendation ID
            user_id: Optional user ID for filtering
            
        Returns:
            Recommendation if found, None otherwise
        """
        rec_ref = self.db.collection(self.recommendations_collection).document(str(rec_id))
        rec_doc = rec_ref.get()
        
        if not rec_doc.exists:
            return None
            
        recommendation = self._recommendation_from_doc(rec_doc.to_dict())
        
        # Check user ID if provided
        if user_id and recommendation.user_id != user_id:
            return None
        
        # Get associated tags and category for response
        tags = []
        if recommendation.tag_ids:
            for tag_id in recommendation.tag_ids:
                tag = await self.get_tag(tag_id)
                if tag:
                    tags.append(tag)
        
        category = None
        if recommendation.category_id:
            category = await self.get_category(recommendation.category_id)
        
        # Return response model
        return Recommendation(
            id=recommendation.id,
            user_id=recommendation.user_id,
            title=recommendation.title,
            content=recommendation.content,
            sources=recommendation.sources,
            metadata=recommendation.metadata,
            tags=tags,
            category=category,
            created_at=recommendation.created_at,
            updated_at=recommendation.updated_at
        )
    
    async def get_recommendations(self, 
                             user_id: UUID, 
                             tag_ids: Optional[List[UUID]] = None,
                             category_id: Optional[UUID] = None,
                             search_query: Optional[str] = None,
                             limit: int = 100,
                             offset: int = 0) -> List[Recommendation]:
        """
        Get recommendations with optional filtering.
        
        Args:
            user_id: User ID
            tag_ids: Optional list of tag IDs to filter by
            category_id: Optional category ID to filter by
            search_query: Optional search query for title/content
            limit: Maximum results to return
            offset: Offset for pagination
            
        Returns:
            List of recommendations
        """
        query = self.db.collection(self.recommendations_collection).where("user_id", "==", str(user_id))
        
        # Apply category filter if provided
        if category_id:
            query = query.where("category_id", "==", str(category_id))
            
        # Get all potential matches (we'll filter for tags manually since Firestore has limits on composite queries)
        docs = list(query.stream())
        
        results = []
        for doc in docs:
            recommendation = self._recommendation_from_doc(doc.to_dict())
            
            # Filter by tags if needed
            if tag_ids:
                # Skip if not all specified tags are present
                if not all(tid in recommendation.tag_ids for tid in tag_ids):
                    continue
                    
            # Simple text search if needed
            if search_query:
                search_terms = search_query.lower().split()
                text_content = f"{recommendation.title} {recommendation.content}".lower()
                if not all(term in text_content for term in search_terms):
                    continue
            
            # Get associated tags and category for response
            tags = []
            if recommendation.tag_ids:
                for tag_id in recommendation.tag_ids:
                    tag = await self.get_tag(tag_id)
                    if tag:
                        tags.append(tag)
            
            category = None
            if recommendation.category_id:
                category = await self.get_category(recommendation.category_id)
            
            # Add to results
            results.append(Recommendation(
                id=recommendation.id,
                user_id=recommendation.user_id,
                title=recommendation.title,
                content=recommendation.content,
                sources=recommendation.sources,
                metadata=recommendation.metadata,
                tags=tags,
                category=category,
                created_at=recommendation.created_at,
                updated_at=recommendation.updated_at
            ))
        
        # Sort by creation date (newest first)
        results.sort(key=lambda r: r.created_at, reverse=True)
        
        # Apply pagination
        return results[offset:offset+limit]
    
    async def update_recommendation(self, 
                               rec_id: UUID, 
                               user_id: UUID, 
                               data: RecommendationUpdate) -> Optional[Recommendation]:
        """
        Update a recommendation.
        
        Args:
            rec_id: Recommendation ID
            user_id: User ID (for authorization)
            data: Update data
            
        Returns:
            Updated recommendation if successful, None otherwise
        """
        rec_ref = self.db.collection(self.recommendations_collection).document(str(rec_id))
        rec_doc = rec_ref.get()
        
        if not rec_doc.exists:
            return None
            
        # Get existing recommendation
        existing_rec = self._recommendation_from_doc(rec_doc.to_dict())
        
        # Check ownership
        if existing_rec.user_id != user_id:
            return None
            
        # Update fields if provided
        update_data = {}
        
        if data.title is not None:
            update_data["title"] = data.title
            
        if data.content is not None:
            update_data["content"] = data.content
            
        if data.sources is not None:
            update_data["sources"] = data.sources
            
        if data.metadata is not None:
            update_data["metadata"] = data.metadata
            
        if data.tag_ids is not None:
            update_data["tag_ids"] = [str(tag_id) for tag_id in data.tag_ids]
            
        if data.category_id is not None:
            update_data["category_id"] = str(data.category_id) if data.category_id else None
            
        # Update timestamp
        update_data["updated_at"] = firestore.SERVER_TIMESTAMP
        
        # Save changes
        rec_ref.update(update_data)
        
        # Get updated document
        return await self.get_recommendation(rec_id, user_id)
    
    async def delete_recommendation(self, rec_id: UUID, user_id: UUID) -> bool:
        """
        Delete a recommendation.
        
        Args:
            rec_id: Recommendation ID
            user_id: User ID (for authorization)
            
        Returns:
            True if deleted, False otherwise
        """
        rec_ref = self.db.collection(self.recommendations_collection).document(str(rec_id))
        rec_doc = rec_ref.get()
        
        if not rec_doc.exists:
            return False
            
        # Check ownership
        rec_data = rec_doc.to_dict()
        if rec_data.get("user_id") != str(user_id):
            return False
            
        # Delete the recommendation
        rec_ref.delete()
        return True 