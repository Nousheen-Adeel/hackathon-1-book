# Physical AI & Humanoid Robotics - RAG System

Complete Retrieval-Augmented Generation (RAG) system for the "Physical AI & Humanoid Robotics" textbook.

## Overview

This system consists of two main components:
1. **Backend**: Python FastAPI application with document indexing and Qwen-powered chat
2. **Frontend**: Simple web interface for interacting with the RAG system

## Architecture

```
rag_backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── data_ingestion/        # Document loading and chunking
│   └── loader.py
├── embeddings/            # Embedding generation
│   └── generator.py
├── vector_db/             # Qdrant vector database interface
│   └── qdrant.py
└── api/                   # API endpoints and RAG service
    └── rag_service.py

rag_frontend/
├── index.html             # Main web interface
└── README.md              # Frontend documentation
```

## Setup Instructions

### Backend Setup

1. Install Python dependencies:
```bash
cd rag_backend
pip install -r requirements.txt
```

2. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

3. Run the backend server:
```bash
python main.py
# Server will start on http://localhost:8000
```

### Frontend Setup

1. Simply open `rag_frontend/index.html` in your web browser
2. Ensure the backend server is running on `http://localhost:8000`

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key for accessing Qwen models
- `CHAT_MODEL`: Model to use for chat (default: `qwen/qwen-2-72b-instruct`)
- `EMBEDDING_MODEL`: Model to use for embeddings (default: `nvidia/nv-embed-v1`)
- `QDRANT_URL`: URL for Qdrant vector database (default: `http://localhost:6333`)
- `QDRANT_API_KEY`: API key for Qdrant (if using cloud version)
- `QDRANT_COLLECTION_NAME`: Name of the collection to store embeddings (default: `book_docs`)

### Document Sources

The system automatically loads documents from `../../book/docs/` relative to the backend directory. Ensure your textbook content is available in this location as markdown files.

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Main chat endpoint
  - Request: `{"query": "your question", "top_k": 5}`
  - Response: `{"answer": "the answer", "sources": ["source1", "source2"], "query": "your question"}`

## Usage Notes

- On first run, the system will index all documents from the book/docs directory
- Subsequent queries will use the indexed information to generate answers
- If an answer is not found in the book, the system responds with "Answer not found in the book."
- The system uses cosine similarity for document retrieval from the vector database

## Troubleshooting

- If documents don't load, verify the `book/docs/` path is accessible
- Check that your OpenRouter API key is valid and has sufficient credits
- Ensure Qdrant is running if using a local instance
- Check logs in the backend for detailed error information