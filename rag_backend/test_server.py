from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Physical AI & Humanoid Robotics RAG API",
    description="RAG system for answering questions about Physical AI & Humanoid Robotics textbook",
    version="1.0.0"
)

# Import after setting up logging and environment
from api.rag_service import RAGService

# Initialize the RAG service
rag_service = RAGService()

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    query: str

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that processes user queries using RAG
    """
    try:
        result = await rag_service.get_answer(request.query, request.top_k)
        return ChatResponse(**result)
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/docs/sources")
def get_document_sources():
    """Get list of all document sources in the vector database"""
    return {"sources": [], "message": "Source listing not implemented in this version"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Don't run initialization on startup to avoid connection errors
    uvicorn.run(app, host=host, port=port)