"""
Service for exporting recommendations to different formats.
"""
import os
import tempfile
from typing import List, Dict, Any, Optional, Union, BinaryIO, TextIO
from io import BytesIO, StringIO
from uuid import UUID
from pathlib import Path

from wise_nutrition.models.recommendation import Recommendation, Tag, Category

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class ExportService:
    """Service for exporting recommendations to different formats."""
    
    def __init__(self):
        """Initialize the export service."""
        if not PDF_AVAILABLE:
            print("Warning: fpdf not installed. PDF exports will be unavailable.")
    
    async def export_to_text(self, recommendation: Recommendation) -> str:
        """
        Export a recommendation to text format.
        
        Args:
            recommendation: The recommendation to export
            
        Returns:
            Recommendation as formatted text
        """
        output = []
        
        # Add title
        output.append(f"# {recommendation.title}\n")
        
        # Add metadata
        output.append(f"Date: {recommendation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add tags
        if recommendation.tags:
            tags_str = ", ".join(tag.name for tag in recommendation.tags)
            output.append(f"Tags: {tags_str}")
        
        # Add category
        if recommendation.category:
            output.append(f"Category: {recommendation.category.name}")
        
        # Add content
        output.append("\n## Content\n")
        output.append(recommendation.content)
        
        # Add sources
        if recommendation.sources:
            output.append("\n## Sources\n")
            for i, source in enumerate(recommendation.sources, 1):
                source_preview = source.get("content_preview", "N/A")
                source_type = source.get("type", "N/A")
                source_name = source.get("name", f"Source {i}")
                
                output.append(f"{i}. {source_name} ({source_type}): {source_preview}")
        
        # Join with newlines
        return "\n".join(output)
    
    async def export_to_pdf(self, recommendation: Recommendation) -> bytes:
        """
        Export a recommendation to PDF format.
        
        Args:
            recommendation: The recommendation to export
            
        Returns:
            PDF as bytes
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF export requires fpdf to be installed. Run `pip install fpdf`.")
        
        # Initialize FPDF object
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, recommendation.title, ln=True)
        
        # Add metadata
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Date: {recommendation.created_at.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        
        # Add tags
        if recommendation.tags:
            tags_str = ", ".join(tag.name for tag in recommendation.tags)
            pdf.cell(0, 10, f"Tags: {tags_str}", ln=True)
        
        # Add category
        if recommendation.category:
            pdf.cell(0, 10, f"Category: {recommendation.category.name}", ln=True)
        
        # Add content section header
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Content", ln=True)
        
        # Add content
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, recommendation.content)
        
        # Add sources
        if recommendation.sources:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Sources", ln=True)
            
            pdf.set_font("Arial", "", 10)
            for i, source in enumerate(recommendation.sources, 1):
                source_preview = source.get("content_preview", "N/A")
                source_type = source.get("type", "N/A")
                source_name = source.get("name", f"Source {i}")
                
                source_text = f"{i}. {source_name} ({source_type}): {source_preview}"
                pdf.multi_cell(0, 10, source_text)
        
        # Return PDF as bytes
        return pdf.output(dest="S").encode("latin-1")
    
    async def export_multiple_to_text(self, recommendations: List[Recommendation]) -> str:
        """
        Export multiple recommendations to a single text file.
        
        Args:
            recommendations: List of recommendations to export
            
        Returns:
            Recommendations as formatted text
        """
        output = []
        
        output.append("# Nutrition Recommendations\n")
        output.append(f"Total: {len(recommendations)} recommendations\n")
        
        for i, recommendation in enumerate(recommendations, 1):
            output.append(f"\n{'=' * 50}\n")
            output.append(f"## {i}. {recommendation.title}\n")
            
            # Add metadata
            output.append(f"Date: {recommendation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add tags
            if recommendation.tags:
                tags_str = ", ".join(tag.name for tag in recommendation.tags)
                output.append(f"Tags: {tags_str}")
            
            # Add category
            if recommendation.category:
                output.append(f"Category: {recommendation.category.name}")
            
            # Add content
            output.append("\nContent:\n")
            output.append(recommendation.content)
            
            # Add sources (abbreviated for multiple export)
            if recommendation.sources:
                sources_count = len(recommendation.sources)
                output.append(f"\nSources: {sources_count} source(s)")
        
        # Join with newlines
        return "\n".join(output)
    
    async def export_multiple_to_pdf(self, recommendations: List[Recommendation]) -> bytes:
        """
        Export multiple recommendations to a single PDF file.
        
        Args:
            recommendations: List of recommendations to export
            
        Returns:
            PDF as bytes
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF export requires fpdf to be installed. Run `pip install fpdf`.")
        
        # Initialize FPDF object
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Nutrition Recommendations", ln=True)
        
        # Add summary
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Total: {len(recommendations)} recommendations", ln=True)
        
        for i, recommendation in enumerate(recommendations, 1):
            # Add separator
            pdf.ln(10)
            pdf.cell(0, 0, "_" * 80, ln=True)
            pdf.ln(5)
            
            # Add recommendation title
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"{i}. {recommendation.title}", ln=True)
            
            # Add metadata
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"Date: {recommendation.created_at.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            
            # Add tags
            if recommendation.tags:
                tags_str = ", ".join(tag.name for tag in recommendation.tags)
                pdf.cell(0, 10, f"Tags: {tags_str}", ln=True)
            
            # Add category
            if recommendation.category:
                pdf.cell(0, 10, f"Category: {recommendation.category.name}", ln=True)
            
            # Add content
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Content:", ln=True)
            
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 10, recommendation.content)
            
            # Add abbreviated sources info
            if recommendation.sources:
                sources_count = len(recommendation.sources)
                pdf.cell(0, 10, f"Sources: {sources_count} source(s)", ln=True)
        
        # Return PDF as bytes
        return pdf.output(dest="S").encode("latin-1") 