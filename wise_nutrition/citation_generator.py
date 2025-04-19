"""
Citation Generation System for Nutrition RAG.

This module provides functionality for generating properly formatted citations
from retrieved documents, allowing the RAG system to provide source information
for its responses.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


class Citation(BaseModel):
    """Represents a formatted citation from a source document."""
    
    text: str = Field(..., description="The formatted citation text")
    source_name: Optional[str] = Field(None, description="Name of the source")
    source_url: Optional[str] = Field(None, description="URL of the source")
    date_accessed: Optional[str] = Field(None, description="Date the source was accessed")
    original_content: Optional[str] = Field(None, description="Original content from the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {"arbitrary_types_allowed": True}
    
    def to_display_format(self, style: str = "mla") -> str:
        """
        Convert the citation to a formatted display string.
        
        Args:
            style: Citation style to use (mla, apa, chicago)
            
        Returns:
            Formatted citation string
        """
        if style == "mla":
            return self._to_mla_format()
        elif style == "apa":
            return self._to_apa_format()
        elif style == "chicago":
            return self._to_chicago_format()
        else:
            return self.text
    
    def _to_mla_format(self) -> str:
        """Format citation in MLA style."""
        # If text is already provided, return it
        if self.text and "Accessed" in self.text:
            return self.text
            
        source = self.source_name or "Unknown Source"
        url = f", {self.source_url}" if self.source_url else ""
        date = f", Accessed {self.date_accessed}" if self.date_accessed else ""
        
        return f'"{source}"{url}{date}.'
    
    def _to_apa_format(self) -> str:
        """Format citation in APA style."""
        # Don't use pre-set text for other formats
        source = self.source_name or "Unknown Source"
        url = f". Retrieved from {self.source_url}" if self.source_url else ""
        date = f" on {self.date_accessed}" if self.date_accessed else ""
        
        return f"{source}{url}{date}."
    
    def _to_chicago_format(self) -> str:
        """Format citation in Chicago style."""
        # Don't use pre-set text for other formats
        source = self.source_name or "Unknown Source"
        url = f", {self.source_url}" if self.source_url else ""
        date = f", accessed {self.date_accessed.lower()}" if self.date_accessed else ""
        
        return f'"{source}"{url}{date}.'


class CitationGenerator(BaseModel):
    """
    System for generating properly formatted citations from documents.
    
    This component takes retrieved documents and generates citations that
    can be included with RAG responses to provide source attribution.
    """
    
    default_style: str = Field(default="mla", description="Default citation style")
    
    model_config = {"arbitrary_types_allowed": True}
    
    def generate_citation(self, document: Document) -> Citation:
        """
        Generate a citation from a document.
        
        Args:
            document: The document to generate a citation for
            
        Returns:
            A Citation object
        """
        # Extract metadata
        metadata = document.metadata if hasattr(document, "metadata") else {}
        
        # Get source information
        source_name = metadata.get("source") or metadata.get("name") or "Unknown Source"
        source_url = metadata.get("url") or None
        
        # Set access date to today
        date_accessed = datetime.now().strftime("%d %B %Y")
        
        # Extract a snippet of the original content
        original_content = document.page_content if hasattr(document, "page_content") else None
        if original_content and len(original_content) > 100:
            original_content = original_content[:97] + "..."
        
        # Build the formatted citation text based on the default style
        if self.default_style == "apa":
            citation_text = self._build_apa_citation(source_name, source_url, date_accessed, metadata)
        elif self.default_style == "chicago":
            citation_text = self._build_chicago_citation(source_name, source_url, date_accessed, metadata)
        else:  # Default to MLA
            citation_text = self._build_mla_citation(source_name, source_url, date_accessed, metadata)
        
        # Create and return the Citation object
        return Citation(
            text=citation_text,
            source_name=source_name,
            source_url=source_url,
            date_accessed=date_accessed,
            original_content=original_content,
            metadata=metadata
        )
    
    def _build_mla_citation(
        self,
        source_name: str,
        source_url: Optional[str] = None,
        date_accessed: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build an MLA-style citation."""
        url_part = f", {source_url}" if source_url else ""
        date_part = f", Accessed {date_accessed}" if date_accessed else ""
        return f'"{source_name}"{url_part}{date_part}.'
    
    def _build_apa_citation(
        self,
        source_name: str,
        source_url: Optional[str] = None,
        date_accessed: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build an APA-style citation."""
        url_part = f". Retrieved from {source_url}" if source_url else ""
        date_part = f" on {date_accessed}" if date_accessed else ""
        return f"{source_name}{url_part}{date_part}."
    
    def _build_chicago_citation(
        self,
        source_name: str,
        source_url: Optional[str] = None,
        date_accessed: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a Chicago-style citation."""
        url_part = f", {source_url}" if source_url else ""
        date_part = f", accessed {date_accessed}" if date_accessed else ""
        return f'"{source_name}"{url_part}{date_part}.'
    
    def _build_citation_text(
        self,
        source_name: str,
        source_url: Optional[str] = None,
        date_accessed: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a formatted citation text string based on the default style.
        
        Args:
            source_name: Name of the source
            source_url: URL of the source
            date_accessed: Date the source was accessed
            metadata: Additional metadata
            
        Returns:
            Formatted citation text
        """
        if self.default_style == "apa":
            return self._build_apa_citation(source_name, source_url, date_accessed, metadata)
        elif self.default_style == "chicago":
            return self._build_chicago_citation(source_name, source_url, date_accessed, metadata)
        else:  # Default to MLA
            return self._build_mla_citation(source_name, source_url, date_accessed, metadata)
    
    def generate_citations(self, documents: List[Document]) -> List[Citation]:
        """
        Generate citations for multiple documents.
        
        Args:
            documents: List of documents to generate citations for
            
        Returns:
            List of Citation objects
        """
        return [self.generate_citation(doc) for doc in documents]
    
    def as_runnable(self) -> RunnableLambda:
        """Convert this citation generator to a runnable lambda."""
        
        def _generate_citations(inputs: Dict[str, Any]) -> List[Citation]:
            """Generate citations from input documents."""
            documents = inputs.get("documents", [])
            return self.generate_citations(documents)
        
        return RunnableLambda(_generate_citations) 