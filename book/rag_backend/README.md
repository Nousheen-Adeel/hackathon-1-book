# RAG Chatbot for Docusaurus Book

This repository contains a Retrieval-Augmented Generation (RAG) chatbot implementation for the Physical AI & Humanoid Robotics book, integrated with Docusaurus.

## Features

- **Full Book RAG**: Ask questions about the entire book content
- **Strict Context Mode**: Get answers based only on selected text
- **Docusaurus Integration**: Seamless integration with existing Docusaurus setup
- **Qwen LLM**: Using OpenRouter's Qwen model for responses
- **Vector Database**: Qdrant for efficient document retrieval

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js and npm
- OpenRouter API key
- Qdrant Cloud account (free tier)

### Installation

1. Clone or integrate this code with your Docusaurus book
2. Install backend dependencies:
   ```bash
   cd rag_backend
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Ingest book content into vector database:
   ```bash
   python ingest_docs.py
   ```

5. Start the backend server:
   ```bash
   # On Windows
   start_backend.bat
   # Or on Unix
   chmod +x start_backend.sh
   ./start_backend.sh
   ```

6. Start your Docusaurus site:
   ```bash
   npm run start
   ```

## Architecture

### Backend Structure
```
rag_backend/
├── api/                 # FastAPI endpoints
├── data_ingestion/      # Markdown parsing
├── embeddings/          # OpenRouter embedding client
├── vector_db/           # Qdrant integration
├── main.py             # FastAPI application
├── ingest_docs.py      # Document ingestion script
└── requirements.txt    # Python dependencies
```

### Frontend Integration
```
src/
├── components/
│   ├── Chatbot.js      # Chat interface
│   ├── Chatbot.css     # Styling
│   └── ChatbotAPI.js   # API utilities
└── theme/
    └── Layout.js       # Docusaurus theme override
```

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Full book RAG chat
- `POST /chat/selected` - Strict context mode chat
- `GET /health` - Health check

## Environment Variables

Create a `.env` file in the `rag_backend/` directory:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Qdrant Vector Database Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_embeddings

# Application Configuration
EMBEDDING_MODEL=qwen/qwen-2-7b-instruct:free
MAX_TOKENS_PER_CHUNK=512
```

## Production Deployment

1. Deploy the FastAPI backend to a cloud platform
2. Update the frontend to point to your deployed backend URL
3. Ensure all environment variables are properly configured in production

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.