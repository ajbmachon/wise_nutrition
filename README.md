# Wise Nutrition

A RAG-based nutrition advisor using LangChain, FastAPI, and Weaviate/ChromaDB.

## Description

Wise Nutrition is a Python package that provides nutrition advice based on documents processed through a Retrieval-Augmented Generation (RAG) pipeline. It leverages modern LangChain features with pipe syntax, Weaviate or ChromaDB for vector storage, and FastAPI for service delivery.

## Features

- PDF document loading and parsing
- Document embedding and storage in Weaviate or ChromaDB
- Custom retriever for relevant context fetching
- RAG pipeline with conversational memory
- FastAPI-based RESTful API

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wise_nutrition.git
cd wise_nutrition

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# For development with ChromaDB (optional)
pip install -e ".[dev]"
```

## Usage

### Running the API

```bash
uvicorn wise_nutrition.api.main:app --reload
```

The API will be available at http://localhost:8000.

### Vector Database Configuration

The project supports both Weaviate and ChromaDB as vector databases. You can configure which one to use by setting the `VECTOR_DB_TYPE` environment variable:

```bash
# Use Weaviate (default)
export VECTOR_DB_TYPE=weaviate

# Use ChromaDB (for local development)
export VECTOR_DB_TYPE=chroma
```

### ChromaDB Example

For a quick example of using ChromaDB embedding:

```bash
python -m wise_nutrition.embeddings.chroma_example
```

### API Documentation

Once the API is running, you can access the Swagger documentation at http://localhost:8000/docs.

## Development

### Project Structure

- `wise_nutrition/`: Main package directory
  - `document_loaders/`: PDF loading and processing
  - `embeddings/`: Text embedding generation and vector database management
    - `embedding_manager.py`: Weaviate-based embedding manager
    - `chroma_embedding_manager.py`: ChromaDB-based embedding manager
    - `factory.py`: Factory methods for selecting the appropriate embedding manager
  - `retriever/`: Custom retriever implementations
  - `rag/`: RAG pipeline components
  - `memory/`: Conversation memory components
  - `api/`: FastAPI application
  - `utils/`: Utility functions

### Testing

```bash
pytest
```

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 