# RAG Chatbot Implementation Guide

This document provides a complete guide to deploying and testing the Retrieval-Augmented Generation (RAG) chatbot for your Docusaurus book project.

## Overview

The RAG chatbot provides two main capabilities:
1. **Full Book RAG**: Answers questions using the full book content
2. **Strict Context Mode**: Answers questions only using user-selected text

## Prerequisites

Before deploying the RAG chatbot, ensure you have:

1. **OpenRouter API Key** (using Qwen model)
2. **Qdrant Cloud Account** (free tier)
3. **Python 3.8+** for the backend
4. **Node.js & npm** for the Docusaurus frontend
5. **Existing Docusaurus book project**

## Setup Instructions

### 1. Environment Configuration

1. Create a `.env` file in the `rag_backend/` directory:

```bash
cp rag_backend/.env.example rag_backend/.env
```

2. Update the values in `.env`:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `QDRANT_URL`: Your Qdrant Cloud URL
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `NEON_DB_URL`: Your Neon Postgres connection string (optional)

### 2. Backend Setup

1. Navigate to the rag_backend directory:
```bash
cd rag_backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the vector database with book content:
```bash
python ingest_docs.py
```

This will parse all markdown files in the `docs/` directory and store their embeddings in Qdrant.

4. Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

The backend API will be available at `http://localhost:8000`.

### 3. Frontend Integration

The Docusaurus integration is already set up:

1. The chatbot component is automatically included on all pages via the theme override in `src/theme/Layout.js`

2. The API proxy is configured in `docusaurus.config.js` to forward requests from `/api` to the backend

3. Start your Docusaurus development server:
```bash
npm run start
```

## Testing the Implementation

### 1. Backend Testing

Test the backend endpoints directly:

- Health check: `GET http://localhost:8000/health`
- Full book RAG: `POST http://localhost:8000/chat`
- Strict context mode: `POST http://localhost:8000/chat/selected`

### 2. Frontend Testing

1. Visit your book website
2. Click the floating chat button in the bottom right
3. Test both modes:
   - Regular mode: Ask questions about the book content
   - Strict context mode: Select text on the page, enable strict mode, then ask questions

## API Endpoints

### `/chat` (POST)
- **Purpose**: Full book RAG chat
- **Request**: `{ "message": "your question", "history": [], "temperature": 0.7 }`
- **Response**: `{ "response": "answer", "sources": [...] }`

### `/chat/selected` (POST)
- **Purpose**: Strict context mode RAG chat
- **Request**: `{ "message": "your question", "selected_text": "user selected text", "history": [], "temperature": 0.7 }`
- **Response**: `{ "response": "answer", "sources": [...], "warning": "..." }`

## Deployment

### Production Deployment

1. **Backend**:
   - Deploy the FastAPI app to a cloud platform (Heroku, Railway, etc.)
   - Ensure environment variables are properly set
   - Update the `REACT_APP_BACKEND_URL` in the frontend to point to your deployed backend

2. **Frontend**:
   - Build the Docusaurus site: `npm run build`
   - Deploy to GitHub Pages, Netlify, Vercel, or similar
   - Ensure the proxy in `docusaurus.config.js` points to your deployed backend

### Environment Variables for Production

Update these in your production deployment:

**Backend:**
- `OPENROUTER_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`

**Frontend:**
- `REACT_APP_BACKEND_URL` (URL of your deployed FastAPI backend)

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify backend server is running
   - Check that proxy is properly configured
   - Confirm CORS settings if deploying separately

2. **Embedding Generation Issues**:
   - Verify OpenRouter API key is valid
   - Check rate limits on OpenRouter
   - Confirm Qdrant connection details

3. **Document Parsing Issues**:
   - Ensure docs directory path is correct
   - Verify markdown files have proper structure
   - Check file permissions

### Debugging Tips

1. Check backend logs for error messages
2. Use browser developer tools to inspect API requests
3. Verify environment variables are set correctly
4. Test API endpoints directly with a tool like Postman

## Architecture

### Backend Components
- `data_ingestion/markdown_parser.py`: Parses Docusaurus docs
- `embeddings/openrouter_client.py`: Generates embeddings via OpenRouter
- `vector_db/qdrant_client.py`: Stores and retrieves embeddings
- `api/main.py`: FastAPI server with chat endpoints

### Frontend Components
- `src/components/Chatbot.js`: Main chat interface
- `src/components/Chatbot.css`: Chat UI styling
- `src/components/ChatbotAPI.js`: API communication layer
- `src/theme/Layout.js`: Theme override to include chatbot
- `docusaurus.config.js`: Proxy configuration

## Security Considerations

1. Never commit API keys to version control
2. Use environment variables for sensitive data
3. Implement rate limiting in production
4. Validate and sanitize all user inputs
5. Use HTTPS for production deployments

## Free Tier Limitations

This implementation is designed to work with free tiers:

- OpenRouter: Limited requests per minute
- Qdrant Cloud: Limited storage and requests
- Neon Postgres: Limited connections and storage

Monitor usage and upgrade as needed for high-traffic applications.