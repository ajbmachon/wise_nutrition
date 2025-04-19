"""
Query Reformulation Module for Nutrition RAG System.

This module provides utilities for reformulating user queries to improve 
retrieval accuracy by generating multiple perspectives on the original query.
It uses LLMs to rewrite queries in ways that may capture different aspects 
of the user's information need, particularly for nutrition-related queries.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.language_models import BaseLLM

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parses the output of an LLM call into a list of strings, one per line."""
    
    def parse(self, text: str) -> List[str]:
        """Parse the output text into a list of strings, one per line."""
        lines = text.strip().split("\n")
        # Remove any empty lines and filter out any non-query lines (like numbering)
        clean_lines = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            # Remove numbering prefix if present (e.g., "1. ", "1) ", etc.)
            cleaned_line = line.strip()
            if cleaned_line[0].isdigit() and cleaned_line[1:3] in ['. ', ') ', '- ']:
                cleaned_line = cleaned_line[3:].strip()
            clean_lines.append(cleaned_line)
        
        return clean_lines

# Define the nutrition-focused query reformulation prompt
NUTRITION_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI nutrition expert. Your task is to generate four different versions 
of the given nutrition-related question to improve retrieval of relevant nutrition information.

For the question: "{question}"

Generate four different ways to ask this question, focusing on different aspects such as:
1. Specific nutrients or components involved
2. Health benefits or effects
3. Food sources or dietary considerations
4. Scientific or medical perspective

Make each query detailed and specific to improve search results. Provide these alternative 
questions separated by newlines, without numbering or prefixes.
"""
)

class QueryReformulator:
    """
    Query reformulation system that generates multiple versions of a user query
    to improve retrieval results by considering different perspectives.
    """
    
    def __init__(
        self,
        llm: Runnable,
        prompt: PromptTemplate = NUTRITION_QUERY_PROMPT,
        output_parser: BaseOutputParser = None,
        include_original: bool = True
    ):
        """
        Initialize the query reformulator.
        
        Args:
            llm: A Runnable language model instance.
            prompt: The prompt template to use for query reformulation.
            output_parser: Parser for LLM output.
            include_original: Whether to include the original query in results.
        """
        self.llm = llm
        self.prompt = prompt
        self.output_parser = output_parser or LineListOutputParser()
        self.include_original = include_original
    
    def rewrite_query(self, original_query: str) -> List[str]:
        """
        Rewrite the original query into multiple alternative queries.
        
        Args:
            original_query: The user's original query string.
            
        Returns:
            A list of alternative query strings.
        """
        # Format the prompt with the user's query
        formatted_prompt = self.prompt.format(question=original_query)
        
        # Generate alternative queries using the LLM
        # Use invoke instead of predict for newer LangChain versions
        llm_output = self.llm.invoke(formatted_prompt)
        
        # Handle different output types (string or message)
        if hasattr(llm_output, 'content'):
            # If it's a message object with content attribute
            content = llm_output.content
        else:
            # If it's a string
            content = str(llm_output)
        
        # Parse the output into a list of query strings
        alternative_queries = self.output_parser.parse(content)
        
        # Log the generated queries
        print(f"Generated {len(alternative_queries)} alternative queries for: {original_query}")
        for i, query in enumerate(alternative_queries):
            print(f"  Query {i+1}: {query}")
        
        # Include the original query if specified
        if self.include_original and original_query not in alternative_queries:
            all_queries = [original_query] + alternative_queries
        else:
            all_queries = alternative_queries
            
        return all_queries
    
    def as_runnable(self) -> RunnableLambda:
        """Convert this query reformulator to a runnable lambda."""
        return RunnableLambda(self.rewrite_query) 