"""
Custom retriever implementation.
"""
from typing import List, Dict, Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda
from wise_nutrition.reranker import DocumentReRanker, ReRankingConfig


class NutritionRetriever(BaseRetriever):
    """
    Custom retriever for nutrition-related queries.

    Uses a base retriever and adds domain-specific logic for
    retrieving the most relevant nutrition information.
    Implements domain-specific filtering and reranking.

    Future Considerations (Subtask 2.4):
    - Chunking strategies: Optimal chunking for nutrition texts should be configured
      during data ingestion or potentially adjusted in the base_retriever configuration.
    - Evaluation tools: Separate evaluation scripts/notebooks should be developed to
      benchmark performance using metrics like precision, recall, and nDCG against
      a nutrition-specific test set.
    """

    base_retriever: BaseRetriever
    k: int = 4
    reranker: Optional[DocumentReRanker] = None
    use_reranking: bool = False  # Flag to enable/disable reranking
    # Placeholder for future filter config
    # filter_config: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query, applying domain-specific logic.

        Args:
            query: User query string.
            run_manager: Callback manager for the retriever run.

        Returns:
            List of relevant documents.
        """
        # 1. Retrieve initial documents using the base retriever
        try:
            initial_docs = self.base_retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        except Exception as e:
            print(f"Error retrieving documents from base_retriever: {e}")
            return []

        # 2. Placeholder: Preprocess text if needed
        # processed_docs = [self._preprocess_text(doc) for doc in initial_docs]
        processed_docs = initial_docs # No preprocessing for now

        # 3. Apply domain-specific filtering
        filtered_docs = self._apply_domain_filters(processed_docs, query)
        
        # 4. Apply reranking if enabled and reranker is available
        if self.use_reranking and self.reranker:
            try:
                print(f"Applying reranking to {len(filtered_docs)} documents")
                reranked_docs = self.reranker.rerank(filtered_docs, query)
            except Exception as e:
                print(f"Error during reranking: {e}")
                reranked_docs = filtered_docs  # Fallback to filtered docs without reranking
            final_docs = reranked_docs[:self.k]  # Limit to k documents
        else:
            # Skip reranking, just limit to k documents
            final_docs = filtered_docs[:self.k]

        # Modified print statement to avoid potential unterminated string issues
        if self.use_reranking and self.reranker:
            print(f"Retrieved {len(initial_docs)}, Filtered to {len(filtered_docs)}, Reranked and returning top {len(final_docs)} docs for query: {query}")
        else:
            print(f"Retrieved {len(initial_docs)}, Filtered to {len(filtered_docs)}, Returning top {len(final_docs)} docs for query: {query}")
        
        return final_docs

    def _preprocess_text(self, doc: Document) -> Document:
        """
        Placeholder for nutrition-specific text preprocessing.
        (e.g., handle unit conversions, normalize nutrient names)
        """
        # TODO: Implement text preprocessing logic here
        return doc

    def _apply_domain_filters(self, docs: List[Document], query: str) -> List[Document]:
        """
        Apply domain-specific filters and hybrid retrieval strategies to the documents.
        
        This method implements a flexible hybrid retrieval system that combines:
        1. Semantic search (already performed by base_retriever)
        2. Keyword/term matching
        3. Domain-specific intent detection and filtering
        4. Metadata-based boosting/filtering
        
        Args:
            docs: List of documents to filter
            query: The original query string
            
        Returns:
            Filtered list of documents
        """
        print(f"Applying hybrid retrieval strategy to {len(docs)} docs.")
        
        # Skip if no documents
        if not docs:
            return []
            
        # 1. Query intent detection - classify query into types
        query_intent = self._detect_query_intent(query)
        
        # 2. Apply keyword-based filtering with boost
        keyword_scores = self._score_by_keywords(docs, query)
        
        # 3. Prepare the result with scores
        doc_scores = []
        for i, doc in enumerate(docs):
            # Start with base score (1.0)
            score = 1.0
            
            # Apply keyword score boost
            score *= (1.0 + keyword_scores[i])
            
            # Apply metadata-based boosting
            metadata_boost = self._get_metadata_boost(doc, query_intent)
            score *= (1.0 + metadata_boost)
            
            # Store the document with its score
            doc_scores.append((doc, score))
        
        # 4. Sort by score (descending) and extract documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_docs = [doc for doc, _ in doc_scores]
        
        # Print some info about the filtering
        print(f"Applied hybrid filtering. Top score: {doc_scores[0][1]:.2f} if docs exist")
        
        return filtered_docs
    
    def _detect_query_intent(self, query: str) -> Dict[str, float]:
        """
        Detect the intent of the query to determine the best retrieval strategy.
        
        Returns a dictionary mapping intent types to confidence scores.
        """
        query_lower = query.lower()
        
        # Initialize intent scores
        intents = {
            "nutrient_info": 0.0,      # Looking for info about specific nutrients
            "food_sources": 0.0,        # Looking for food sources of nutrients
            "health_condition": 0.0,    # Question about health condition
            "recipe": 0.0,              # Looking for recipes
            "general_nutrition": 0.0,   # General nutrition questions
            "comparison": 0.0,          # Comparing foods or nutrients
            "dietary_restriction": 0.0,  # Questions about dietary restrictions
        }
        
        # Check for nutrient information intent
        nutrient_terms = ["vitamin", "mineral", "protein", "carbohydrate", "fat", 
                         "omega", "calcium", "iron", "zinc", "magnesium", "potassium"]
        for term in nutrient_terms:
            if term in query_lower:
                intents["nutrient_info"] += 0.3
                break
                
        # Check for food sources intent
        if any(term in query_lower for term in ["source", "food", "contain", "rich in", "high in"]):
            intents["food_sources"] += 0.3
            
        # Check for health condition intent
        health_terms = ["deficiency", "health", "condition", "disease", "symptom", 
                       "prevent", "improve", "boost", "benefit"]
        for term in health_terms:
            if term in query_lower:
                intents["health_condition"] += 0.2
                break
                
        # Check for recipe intent
        if any(term in query_lower for term in ["recipe", "make", "cook", "prepare", "meal"]):
            intents["recipe"] += 0.4
            
        # Check for comparison intent
        if any(term in query_lower for term in ["vs", "versus", "compared to", "difference", "better"]):
            intents["comparison"] += 0.3
            
        # Check for dietary restriction intent
        diet_terms = ["vegan", "vegetarian", "keto", "paleo", "gluten", "lactose", "allergy", 
                     "intolerance", "diet"]
        for term in diet_terms:
            if term in query_lower:
                intents["dietary_restriction"] += 0.3
                break
                
        # General nutrition intent - check explicitly for 'nutrition' term
        if "nutrition" in query_lower or "nutrient" in query_lower or "healthy eating" in query_lower:
            intents["general_nutrition"] += 0.5
        
        # General nutrition is the fallback if no other intents are detected
        if all(score == 0.0 for score in intents.values()):
            intents["general_nutrition"] = 0.5
        
        return intents
    
    def _score_by_keywords(self, docs: List[Document], query: str) -> List[float]:
        """
        Score documents based on keyword matching with the query.
        
        Returns a list of scores, one per document.
        """
        scores = []
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        for doc in docs:
            # Initialize score
            keyword_score = 0.0
            content = doc.page_content.lower()
            
            # Simple term frequency scoring
            for term in query_terms:
                if term in content:
                    # Count occurrences of the term
                    term_count = content.count(term)
                    # Add to score, with diminishing returns for multiple occurrences
                    keyword_score += min(0.2, 0.05 * term_count)
                    
                    # Give extra boost for exact term matches (not substring matches)
                    for word in content.split():
                        if word == term:  # Exact match
                            keyword_score += 0.1
                            break
            
            # Check for exact phrases (higher weight)
            for i in range(len(query_terms) - 1):
                phrase = f"{query_terms[i]} {query_terms[i+1]}"
                if phrase in content:
                    keyword_score += 0.15  # Increased from 0.1
            
            # Give extra weight to title matches (if in metadata)
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                name = doc.metadata.get('name', '').lower()
                if name:
                    for term in query_terms:
                        if term in name:
                            keyword_score += 0.3  # Title match is valuable
            
            scores.append(keyword_score)
        
        return scores
    
    def _get_metadata_boost(self, doc: Document, query_intent: Dict[str, float]) -> float:
        """
        Calculate boost based on document metadata and query intent.
        
        Returns a boost value to multiply with the document's score.
        """
        # Start with no boost
        boost = 0.0
        
        # If no metadata, return no boost
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            return boost
            
        # Get document type from metadata
        doc_type = doc.metadata.get('type', '').lower()
        
        # Apply boosts based on intent and doc type matches
        if doc_type == 'vitamin' and query_intent["nutrient_info"] > 0:
            boost += 0.3
        elif doc_type == 'mineral' and query_intent["nutrient_info"] > 0:
            boost += 0.3
        elif doc_type == 'recipe' and query_intent["recipe"] > 0:
            boost += 0.4
        elif doc_type == 'diet_advice' and query_intent["general_nutrition"] > 0:
            boost += 0.2
        
        # Boost authoritative sources for health-related queries
        source = doc.metadata.get('source', '').lower()
        if query_intent["health_condition"] > 0:
            if any(src in source for src in ["nih.gov", "cdc.gov", "who.int", "mayoclinic"]):
                boost += 0.3
                
        # Recency boost for time-sensitive content
        if 'date' in doc.metadata:
            try:
                doc_date = doc.metadata['date']
                if isinstance(doc_date, str):
                    from datetime import datetime
                    doc_date = datetime.fromisoformat(doc_date)
                    # Simple recency calculation (more recent = higher boost)
                    age_days = (datetime.now() - doc_date).days
                    if age_days < 365:  # Less than a year old
                        boost += 0.1
            except (ValueError, TypeError):
                pass
                
        return boost

    # Placeholder for filter matching logic
    def _matches_filters(self, doc: Document, query: str) -> bool:
        """Logic to check if doc matches filters based on query context."""
        return True

    def as_runnable(self) -> RunnableLambda:
        """
        Convert this retriever to a runnable lambda for use in LCEL chains.

        Returns:
            A runnable lambda wrapping the retriever logic.
        """
        return RunnableLambda(self.get_relevant_documents)

    @classmethod
    def with_reranker(
        cls,
        base_retriever: BaseRetriever,
        k: int = 4,
        reranking_config: Optional[ReRankingConfig] = None,
        **kwargs
    ) -> "NutritionRetriever":
        """
        Create a NutritionRetriever with reranking capabilities.
        
        Args:
            base_retriever: Base retriever to use for document retrieval
            k: Number of documents to return
            reranking_config: Optional custom configuration for reranking
            
        Returns:
            A NutritionRetriever with reranking enabled
        """
        # Create reranker with provided config or default
        reranker = DocumentReRanker(config=reranking_config)
        
        # Create and return retriever with reranking enabled
        return cls(
            base_retriever=base_retriever,
            k=k,
            reranker=reranker,
            use_reranking=True,
            **kwargs
        ) 