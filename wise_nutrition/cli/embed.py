"""
CLI command for embedding files into ChromaDB.
"""
import os
import click
import asyncio
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from wise_nutrition.utils.config import Config
from wise_nutrition.embeddings.chroma_embedding_manager import ChromaEmbeddingManager

# TODO: Bugfix this file in regards to actual data. Also unify the data in regards to the fields as much as possible!

@click.group()
def embed():
    """Embed files into ChromaDB collections."""
    pass


@embed.command()
@click.argument('file_path', type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path))
@click.option('--collection-name', '-c', default=None, help='Name of the collection to create or use')
@click.option('--persist-dir', '-d', default=None, help='Directory to persist ChromaDB data')
@click.option('--clear-existing/--no-clear-existing', default=False, help='Clear existing collection if it exists')
@click.option('--async-mode/--sync-mode', default=True, help='Use async or sync mode for operations')
def files(file_path: Path, collection_name: Optional[str], persist_dir: Optional[str], 
         clear_existing: bool, async_mode: bool):
    """
    Embed files into ChromaDB.
    
    FILE_PATH can be a single file or directory containing files to embed.
    """
    if async_mode:
        asyncio.run(_embed_files_async(file_path, collection_name, persist_dir, clear_existing))
    else:
        _embed_files_sync(file_path, collection_name, persist_dir, clear_existing)


async def _embed_files_async(file_path: Path, collection_name: Optional[str], 
                            persist_dir: Optional[str], clear_existing: bool):
    """Async implementation of file embedding."""
    # Set up embedding manager
    config = Config()
    
    # Override defaults if specified
    if collection_name:
        click.echo(f"Using collection name: {collection_name}")
    else:
        collection_name = config.chroma_collection_name
        click.echo(f"Using default collection name: {collection_name}")
    
    if persist_dir:
        click.echo(f"Using persist directory: {persist_dir}")
    else:
        persist_dir = config.chroma_persist_directory
        click.echo(f"Using default persist directory: {persist_dir}")
    
    # Create embedding manager
    embedding_manager = ChromaEmbeddingManager(
        config=config,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    # Create or reset collection if needed
    if clear_existing:
        click.echo(f"Creating or resetting collection: {collection_name}")
        await embedding_manager.create_collection()
    
    # Process and embed files
    documents = await load_documents(file_path)
    if documents:
        click.echo(f"Adding {len(documents)} documents to collection")
        await embedding_manager.add_documents(documents)
        click.echo("Documents successfully embedded")
    else:
        click.echo("No documents to embed")


def _embed_files_sync(file_path: Path, collection_name: Optional[str], 
                     persist_dir: Optional[str], clear_existing: bool):
    """Sync implementation of file embedding."""
    # Set up embedding manager
    config = Config()
    
    # Override defaults if specified
    if collection_name:
        click.echo(f"Using collection name: {collection_name}")
    else:
        collection_name = config.chroma_collection_name
        click.echo(f"Using default collection name: {collection_name}")
    
    if persist_dir:
        click.echo(f"Using persist directory: {persist_dir}")
    else:
        persist_dir = config.chroma_persist_directory
        click.echo(f"Using default persist directory: {persist_dir}")
    
    # Create embedding manager
    embedding_manager = ChromaEmbeddingManager(
        config=config,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    # Create or reset collection if needed
    if clear_existing:
        click.echo(f"Creating or resetting collection: {collection_name}")
        embedding_manager.create_collection_sync()
    
    # Process and embed files
    documents = load_documents_sync(file_path)
    if documents:
        click.echo(f"Adding {len(documents)} documents to collection")
        embedding_manager.add_documents_sync(documents)
        click.echo("Documents successfully embedded")
    else:
        click.echo("No documents to embed")


async def load_documents(file_path: Path) -> List[Document]:
    """
    Load documents from file or directory.
    
    Delegates to sync version for now, but allows for future async file loading.
    """
    return load_documents_sync(file_path)


def load_documents_sync(file_path: Path) -> List[Document]:
    """
    Load documents from file or directory.
    
    Currently supports simple text processing.
    Can be extended for specific file formats (JSON, CSV, etc).
    """
    documents = []
    
    if file_path.is_file():
        # Process single file
        click.echo(f"Processing file: {file_path}")
        documents.extend(_process_file(file_path))
    elif file_path.is_dir():
        # Process all files in directory
        click.echo(f"Processing directory: {file_path}")
        for root, _, files in os.walk(file_path):
            for filename in files:
                filepath = Path(root) / filename
                click.echo(f"Processing file: {filepath}")
                documents.extend(_process_file(filepath))
    
    return documents


def _process_file(file_path: Path) -> List[Document]:
    """
    Process a single file based on its extension.
    
    Can be extended to support different file formats.
    """
    # Simple implementation - check file extension and handle accordingly
    suffix = file_path.suffix.lower()
    
    if suffix == '.txt':
        return _process_text_file(file_path)
    elif suffix == '.json':
        return _process_json_file(file_path)
    elif suffix == '.csv':
        return _process_csv_file(file_path)
    else:
        click.echo(f"Unsupported file format: {suffix}")
        return []


def _process_text_file(file_path: Path) -> List[Document]:
    """Process a text file into a Document."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a document with the content and metadata
        return [Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "chunk_id": f"file_{file_path.stem}"
            }
        )]
    except Exception as e:
        click.echo(f"Error processing {file_path}: {e}")
        return []


def _process_json_file(file_path: Path) -> List[Document]:
    """Process a JSON file into Documents."""
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Assume list of items (like recipes, nutrition facts, etc.)
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Extract content and metadata fields
                    content = _extract_content_from_dict(item)
                    metadata = {k: v for k, v in item.items() if k != 'content'}
                    
                    # Add file source and chunk_id to metadata
                    metadata.update({
                        "source": str(file_path),
                        "filename": file_path.name,
                        "chunk_id": f"json_{file_path.stem}_{i}"
                    })
                    
                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
        elif isinstance(data, dict):
            # Single document
            content = _extract_content_from_dict(data)
            metadata = {k: v for k, v in data.items() if k != 'content'}
            
            # Add file source and chunk_id to metadata
            metadata.update({
                "source": str(file_path),
                "filename": file_path.name,
                "chunk_id": f"json_{file_path.stem}"
            })
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents
    except Exception as e:
        click.echo(f"Error processing {file_path}: {e}")
        return []


# TODO: This needs to be corrected, our main values currently live in quote and instructions fields. Unify the data or update this method accordingly.
def _extract_content_from_dict(data: dict) -> str:
    """
    Extract content from a dictionary representation.
    
    Rules:
    1. Use 'content' field if available
    2. Otherwise, concatenate available text fields
    """
    # If there's a 'content' field, use it
    if 'content' in data:
        return str(data['content'])
    
    # Otherwise, try some common fields that might contain text
    potential_content_fields = [
        'description', 'text', 'body', 'summary', 'info',
        'details', 'title', 'name', 'instructions'
    ]
    
    content_parts = []
    for field in potential_content_fields:
        if field in data and data[field]:
            if field in ['title', 'name']:
                content_parts.insert(0, f"{data[field]}\n")
            else:
                content_parts.append(str(data[field]))
    
    # If we found content fields, join them
    if content_parts:
        return "\n\n".join(content_parts)
    
    # As a last resort, serialize the whole object
    return str(data)


def _process_csv_file(file_path: Path) -> List[Document]:
    """Process a CSV file into Documents."""
    import csv
    
    try:
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Extract row data
                content = _extract_content_from_dict(row)
                
                # Create metadata
                metadata = {k: v for k, v in row.items()}
                metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_id": f"csv_{file_path.stem}_{i}"
                })
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        return documents
    except Exception as e:
        click.echo(f"Error processing {file_path}: {e}")
        return []


if __name__ == "__main__":
    embed() 