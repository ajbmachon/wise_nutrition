# Wise Nutrition

A RAG-based nutrition advisor using LangChain, FastAPI, and Weaviate.

## Description

Wise Nutrition is a Python package that provides nutrition advice based on documents processed through a Retrieval-Augmented Generation (RAG) pipeline. It leverages modern LangChain features with pipe syntax, Weaviate for vector storage, and FastAPI for service delivery.

## Features

- PDF document loading and parsing
- Document embedding and storage in Weaviate
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
```

## Usage

### Running the API

```bash
uvicorn wise_nutrition.api.main:app --reload
```

The API will be available at http://localhost:8000.

### API Documentation

Once the API is running, you can access the Swagger documentation at http://localhost:8000/docs.

## Development

### Project Structure

- `wise_nutrition/`: Main package directory
  - `document_loaders/`: PDF loading and processing
  - `embeddings/`: Text embedding generation
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