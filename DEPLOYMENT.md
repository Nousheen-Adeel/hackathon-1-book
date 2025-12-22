# Deployment Guide

## Railway Deployment Setup

This project is configured for deployment on Railway via GitHub integration.

### Tech Stack
- **Backend**: Python FastAPI RAG system in `rag_backend/`
- **Frontend**: Docusaurus documentation site in `docusaurus_project_new/`

### Backend Service (`rag_backend`)
- Built with Python 3.11 and FastAPI
- Exposes a `/chat` endpoint for AI interactions
- Connects to Qdrant vector database

### Frontend Service (`docusaurus_project_new`)
- Built with Docusaurus
- Static site that consumes the backend API

## Required Environment Variables

Add these environment variables to your Railway project:

### Backend Service
- `OPENROUTER_API_KEY` - Your OpenRouter API key
- `QDRANT_API_KEY` - Your Qdrant API key (if using cloud)
- `QDRANT_URL` - URL to your Qdrant instance
- `QDRANT_COLLECTION_NAME` - Name of your Qdrant collection
- `CHAT_MODEL` - Model to use (default: qwen/qwen-2-72b-instruct)
- `EMBEDDING_MODEL` - Embedding model to use (default: nvidia/nv-embed-v1)

### Frontend Service
- `BACKEND_URL` - URL of your deployed backend service

## Deployment Steps

1. **Connect GitHub Repository** to Railway
2. **Create Two Services**:
   - One for `rag_backend/` directory
   - One for `docusaurus_project_new/` directory

3. **Configure Environment Variables** for each service

4. **Deploy** both services

## Files Added for Railway Compatibility

- `Dockerfile` in both `rag_backend/` and `docusaurus_project_new/`
- `Procfile` in both directories
- `railway.toml` in root directory
- Modified `start_server.py` to use PORT environment variable

## Important Notes

- The backend now dynamically reads the PORT environment variable provided by Railway
- Both services are built separately for optimal deployment
- Make sure to set the correct working directory when configuring services in Railway