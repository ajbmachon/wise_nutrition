"""
Post-Retrieval Re-ranking System for Nutrition Documents.

This module provides functionality to re-rank retrieved documents based on
sophisticated relevance metrics to improve the quality of results shown to users.
"""
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable, RunnableLambda

class ReRankingConfig(BaseModel):
    """Configuration for document re-ranking."""
    
    # Weight factors for different scoring components
    semantic_weight: float = Field(default=0.6, 
                                   description="Weight for semantic similarity score")
    freshness_weight: float = Field(default=0.1, 
                                    description="Weight for document freshness score")
    authority_weight: float = Field(default=0.15, 
                                    description="Weight for source authority score")
    term_proximity_weight: float = Field(default=0.15, 
                                         description="Weight for term proximity score")
    
    # Configuration for semantic similarity
    use_llm_reranker: bool = Field(default=False, 
                                   description="Whether to use LLM for relevance scoring")
    
    # Configuration for freshness scoring
    max_age_days: int = Field(default=365, 
                              description="Maximum age in days for freshness scoring")
    
    # Configuration for efficiency
    top_n_to_rerank: int = Field(default=20, 
                                 description="Only rerank the top N initial results")
    
    # Nutrition domain specific weights
    nutrient_match_bonus: float = Field(default=0.2, 
                                       description="Bonus for matching specific nutrients")
    
    model_config = {"arbitrary_types_allowed": True}

class DocumentScorer(BaseModel):
    """
    Base class for document scoring components.
    Scoring components calculate relevance scores for documents.
    """
    
    weight: float = Field(default=1.0, description="Weight for this scoring component")
    
    def score_documents(self, documents: List[Document], query: str) -> List[float]:
        """
        Score the documents based on relevance to the query.
        
        Args:
            documents: List of documents to score
            query: The query to score against
            
        Returns:
            List of scores, one per document
        """
        raise NotImplementedError("Subclasses must implement score_documents")
    
    model_config = {"arbitrary_types_allowed": True}

class SemanticSimilarityScorer(DocumentScorer):
    """
    Scores documents based on semantic similarity to the query.
    """
    
    llm: Optional[BaseLLM] = None
    
    def score_documents(self, documents: List[Document], query: str) -> List[float]:
        """Score documents based on semantic similarity to the query."""
        
        # Simple fallback if no LLM is provided
        if not self.llm:
            # Default to basic keyword matching as fallback
            scores = []
            query_terms = query.lower().split()
            for doc in documents:
                content = doc.page_content.lower()
                # Count how many query terms appear in the document
                term_matches = sum(1 for term in query_terms if term in content)
                # Normalize by number of terms
                score = term_matches / max(1, len(query_terms))
                scores.append(score)
            return scores
        
        # TODO: If using LLM for scoring, implement more sophisticated
        # semantic similarity calculation here (e.g., using embeddings)
        
        return [0.5] * len(documents)  # Placeholder

class FreshnessScorer(DocumentScorer):
    """
    Scores documents based on their recency/freshness.
    """
    
    max_age_days: int = 365  # Max age to consider for scoring
    
    def score_documents(self, documents: List[Document], query: str) -> List[float]:
        """Score documents based on their age/freshness."""
        
        scores = []
        current_time = datetime.now()
        
        for doc in documents:
            # Extract date from metadata if available
            doc_date = None
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                date_str = doc.metadata.get('date') or doc.metadata.get('created_at')
                if date_str:
                    try:
                        doc_date = datetime.fromisoformat(date_str)
                    except (ValueError, TypeError):
                        doc_date = None
            
            if doc_date:
                # Calculate age in days
                age_days = (current_time - doc_date).days
                # Score inversely proportional to age, capped at max_age_days
                score = max(0, 1 - (age_days / self.max_age_days))
            else:
                # Default score if no date available
                score = 0.5
            
            scores.append(score)
        
        return scores

class AuthorityScorer(DocumentScorer):
    """
    Scores documents based on source authority/credibility.
    """
    
    authority_sources: Dict[str, float] = Field(default_factory=dict,
                                               description="Mapping of source names to authority scores")
    
    def __init__(self, authority_sources: Optional[Dict[str, float]] = None, **kwargs):
        """Initialize with optional authority sources mapping."""
        
        # Default authority sources for nutrition domains if none provided
        default_authorities = {
            "nih.gov": 0.9,
            "cdc.gov": 0.9,
            "mayoclinic.org": 0.85,
            "harvard.edu": 0.85,
            "who.int": 0.9,
            "nutrition.org": 0.8,
            "nutritionfacts.org": 0.75,
        }
        
        super().__init__(
            authority_sources=authority_sources or default_authorities,
            **kwargs
        )
    
    def score_documents(self, documents: List[Document], query: str) -> List[float]:
        """Score documents based on their source authority."""
        
        scores = []
        
        for doc in documents:
            score = 0.5  # Default middle score
            
            # Check source from metadata
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                source = doc.metadata.get('source')
                url = doc.metadata.get('url', '')
                
                # Check if source is in authority mapping
                if source and source in self.authority_sources:
                    score = self.authority_sources[source]
                # If no direct match, check for domain match in URL
                elif url:
                    for domain, auth_score in self.authority_sources.items():
                        if domain in url:
                            score = auth_score
                            break
            
            scores.append(score)
        
        return scores

class TermProximityScorer(DocumentScorer):
    """
    Scores documents based on the proximity of query terms in the text.
    """
    
    def score_documents(self, documents: List[Document], query: str) -> List[float]:
        """Score documents based on query term proximity."""
        
        scores = []
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        # If query is too short, return neutral scores
        if len(query_terms) < 2:
            return [0.5] * len(documents)
        
        for doc in documents:
            score = 0.5  # Default score
            
            # Simple implementation: check if consecutive query terms appear close together
            content = doc.page_content.lower()
            
            # Count instances where query terms are within a certain window of each other
            windows_found = 0
            window_size = 10  # words
            
            # Convert content to word list for sliding window
            words = content.split()
            
            # Slide window through the document
            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i+window_size])
                # Count how many query terms are in this window
                terms_in_window = sum(1 for term in query_terms if term in window)
                if terms_in_window >= 2:  # At least two query terms in proximity
                    windows_found += 1
            
            # Score based on number of proximity windows found
            if windows_found > 0:
                score = min(1.0, 0.5 + (windows_found * 0.1))
            
            scores.append(score)
        
        return scores

class NutritionSpecificScorer(DocumentScorer):
    """
    Nutrition domain-specific scoring component.
    """
    
    def score_documents(self, documents: List[Document], query: str) -> List[float]:
        """Apply nutrition-specific scoring to documents."""
        
        scores = []
        
        # Extract potential nutrition terms from query
        query_lower = query.lower()
        nutrition_terms = [
            "vitamin", "mineral", "protein", "carbohydrate", "fat", "omega", 
            "calcium", "iron", "zinc", "magnesium", "potassium", "sodium",
            "fiber", "nutrient", "diet", "calorie", "supplement", "deficiency",
            "meal", "nutrition", "food", "health", "metabolism"
        ]
        
        # Check if query contains nutrition terms
        query_nutrition_terms = [term for term in nutrition_terms if term in query_lower]
        
        for doc in documents:
            score = 0.5  # Default score
            
            # If query has nutrition terms, check if document contains the same terms
            if query_nutrition_terms:
                content = doc.page_content.lower()
                matches = sum(1 for term in query_nutrition_terms if term in content)
                if matches > 0:
                    # Boost score based on nutrition term matches
                    score = min(1.0, 0.5 + (matches / len(query_nutrition_terms) * 0.5))
            
            scores.append(score)
        
        return scores

class DocumentReRanker(BaseModel):
    """
    Main reranking system that combines multiple scoring components
    to rerank retrieved documents.
    """
    
    config: ReRankingConfig = Field(default_factory=ReRankingConfig)
    scorers: List[DocumentScorer] = Field(default_factory=list)
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        config: Optional[ReRankingConfig] = None,
        semantic_scorer: Optional[SemanticSimilarityScorer] = None,
        freshness_scorer: Optional[FreshnessScorer] = None,
        authority_scorer: Optional[AuthorityScorer] = None,
        term_proximity_scorer: Optional[TermProximityScorer] = None,
        nutrition_scorer: Optional[NutritionSpecificScorer] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs
    ):
        """
        Initialize the document reranker with scoring components.
        
        Args:
            config: Configuration for reranking
            semantic_scorer: Optional custom semantic similarity scorer
            freshness_scorer: Optional custom freshness scorer
            authority_scorer: Optional custom authority scorer
            term_proximity_scorer: Optional custom term proximity scorer
            nutrition_scorer: Optional custom nutrition domain scorer
            llm: Optional language model for semantic scoring
        """
        # Create config if not provided
        if config is None:
            config = ReRankingConfig()
        
        # Create default scorers if not provided
        scorers = []
        
        # Semantic similarity scorer
        if semantic_scorer is None:
            semantic_scorer = SemanticSimilarityScorer(
                weight=config.semantic_weight,
                llm=llm
            )
        scorers.append(semantic_scorer)
        
        # Freshness scorer
        if freshness_scorer is None:
            freshness_scorer = FreshnessScorer(
                weight=config.freshness_weight,
                max_age_days=config.max_age_days
            )
        scorers.append(freshness_scorer)
        
        # Authority scorer
        if authority_scorer is None:
            authority_scorer = AuthorityScorer(
                weight=config.authority_weight
            )
        scorers.append(authority_scorer)
        
        # Term proximity scorer
        if term_proximity_scorer is None:
            term_proximity_scorer = TermProximityScorer(
                weight=config.term_proximity_weight
            )
        scorers.append(term_proximity_scorer)
        
        # Nutrition domain scorer
        if nutrition_scorer is None:
            nutrition_scorer = NutritionSpecificScorer(
                weight=config.nutrient_match_bonus
            )
        scorers.append(nutrition_scorer)
        
        super().__init__(
            config=config,
            scorers=scorers,
            **kwargs
        )
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank the documents based on the combined scores from all scorers.
        
        Args:
            documents: List of documents to rerank
            query: The original query
            
        Returns:
            Reranked list of documents
        """
        # If no documents or only one, return as-is
        if len(documents) <= 1:
            return documents
        
        # Limit to top N if configured
        docs_to_rerank = documents[:self.config.top_n_to_rerank] if len(documents) > self.config.top_n_to_rerank else documents
        remaining_docs = documents[self.config.top_n_to_rerank:] if len(documents) > self.config.top_n_to_rerank else []
        
        # Skip reranking if no documents to rerank
        if not docs_to_rerank:
            return documents
        
        # Calculate scores from each scorer
        all_scores = []
        for scorer in self.scorers:
            try:
                scores = scorer.score_documents(docs_to_rerank, query)
                # Normalize to ensure weights work as expected
                if scores:
                    max_score = max(scores) if max(scores) > 0 else 1.0
                    normalized_scores = [score / max_score for score in scores]
                    all_scores.append((normalized_scores, scorer.weight))
            except Exception as e:
                print(f"Error in scorer {type(scorer).__name__}: {e}")
        
        # Combine scores using weights
        final_scores = []
        for i in range(len(docs_to_rerank)):
            total_score = sum(scores[i] * weight for scores, weight in all_scores)
            total_weight = sum(weight for _, weight in all_scores)
            combined_score = total_score / total_weight if total_weight > 0 else 0
            final_scores.append(combined_score)
        
        # Create (document, score) pairs for sorting
        doc_score_pairs = list(zip(docs_to_rerank, final_scores))
        
        # Sort by score in descending order
        reranked_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Extract just the documents
        reranked_docs = [doc for doc, _ in reranked_pairs]
        
        # Add any remaining documents that weren't reranked
        if remaining_docs:
            reranked_docs.extend(remaining_docs)
        
        # Print reranking summary
        print(f"Reranked {len(docs_to_rerank)} documents. Top score: {final_scores[doc_score_pairs.index(reranked_pairs[0])]:.2f}")
        
        return reranked_docs
    
    def as_runnable(self) -> Runnable:
        """Convert the reranker to a runnable for use in chains."""
        
        def _rerank_docs(inputs: Dict[str, Any]) -> List[Document]:
            """Inner function for the runnable lambda."""
            documents = inputs.get("documents", [])
            query = inputs.get("query", "")
            
            if not documents or not query:
                return documents
            
            return self.rerank(documents, query)
        
        return RunnableLambda(_rerank_docs) 