"""
PDF document loader module.
"""
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from wise_nutrition.utils.config import Config


class NutritionPDFLoader:
    """
    Load and process nutrition-related PDF documents.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        config: Config = Config()
    ):
        """
        Initialize the PDF loader.
        
        Args:
            chunk_size: The size of text chunks to create
            chunk_overlap: The overlap between consecutive chunks
        """
        self._config = config
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
    def load_and_split(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and split it into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks
        """
        pass
    
    def load_multiple_and_split(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files and split them into chunks.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            List of document chunks from all files
        """
        pass 