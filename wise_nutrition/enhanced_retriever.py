"""
Enhanced Retriever with Query Reformulation capabilities.

This module extends the NutritionRetriever to include query reformulation,
which generates multiple alternative queries to improve retrieval accuracy.
"""
from typing import List, Dict, Any, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda

from wise_nutrition.retriever import NutritionRetriever
from wise_nutrition.query_reformulation import QueryReformulator

class EnhancedNutritionRetriever(NutritionRetriever):
    """
    Enhanced retriever for nutrition-related queries that uses query reformulation
    to improve retrieval accuracy.
    
    This retriever extends the NutritionRetriever by adding query reformulation
    capabilities, which generate multiple perspectives on the original query
    to retrieve a more diverse and comprehensive set of documents.
    """
    
    query_reformulator: Optional[QueryReformulator] = None
    max_queries: int = 4  # Maximum number of alternative queries to use
    use_reformulation: bool = True  # Enable/disable reformulation at runtime
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query using query reformulation.
        
        Args:
            query: Original user query string.
            run_manager: Callback manager for the retriever run.
            
        Returns:
            List of relevant documents.
        """
        if not self.use_reformulation or self.query_reformulator is None:
            # Fall back to base retriever functionality if reformulation is disabled
            return super()._get_relevant_documents(query, run_manager=run_manager)
        
        # Generate alternative queries
        try:
            alternative_queries = self.query_reformulator.rewrite_query(query)
            # Limit to max_queries
            alternative_queries = alternative_queries[:self.max_queries]
        except Exception as e:
            print(f"Error during query reformulation: {e}")
            # Fall back to original query if reformulation fails
            alternative_queries = [query]
        
        # Retrieve documents for each alternative query
        all_docs = []
        for alt_query in alternative_queries:
            try:
                # Use base retriever to get documents for this alternative query
                docs = self.base_retriever.get_relevant_documents(
                    alt_query, callbacks=run_manager.get_child()
                )
                all_docs.extend(docs)
                print(f"Retrieved {len(docs)} docs for query: '{alt_query}'")
            except Exception as e:
                print(f"Error retrieving documents for query '{alt_query}': {e}")
        
        # Remove duplicate documents by page_content
        unique_docs = self._deduplicate_documents(all_docs)
        print(f"Total unique documents after deduplication: {len(unique_docs)}")
        
        # Apply domain-specific filtering as in the base class
        filtered_docs = self._apply_domain_filters(unique_docs, query)
        
        # Limit to k documents
        final_docs = filtered_docs[:self.k]
        
        return final_docs
    
    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on page_content.
        
        Args:
            docs: List of documents to deduplicate.
            
        Returns:
            List of deduplicated documents.
        """
        unique_docs = {}
        for doc in docs:
            # Use page_content as a key for deduplication
            if hasattr(doc, 'page_content'):
                unique_docs[doc.page_content] = doc
        
        return list(unique_docs.values())
    
    @classmethod
    def from_llm(
        cls,
        base_retriever: BaseRetriever,
        llm: Runnable,
        k: int = 4,
        max_queries: int = 4,
        include_original: bool = True,
    ) -> "EnhancedNutritionRetriever":
        """
        Create an EnhancedNutritionRetriever with a query reformulator powered by an LLM.
        
        Args:
            base_retriever: Base retriever to use for document retrieval.
            llm: Language model to use for query reformulation.
            k: Number of documents to return.
            max_queries: Maximum number of alternative queries to use.
            include_original: Whether to include the original query in alternatives.
            
        Returns:
            An EnhancedNutritionRetriever instance.
        """
        # Create a query reformulator with the provided LLM
        query_reformulator = QueryReformulator(llm=llm, include_original=include_original)
        
        # Create and return the enhanced retriever
        return cls(
            base_retriever=base_retriever,
            k=k,
            query_reformulator=query_reformulator,
            max_queries=max_queries,
            use_reformulation=True
        ) 