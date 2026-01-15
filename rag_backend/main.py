from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the RAG service
from api.rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    strict_mode: bool = False

    @property
    def text(self) -> str:
        return self.question or self.query or ""


app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000", "http://127.0.0.1:3001", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG service globally
rag_service = RAGService()

@app.post("/chat")
async def chat(request: QuestionRequest) -> Dict[str, str]:
    """
    Chat endpoint that accepts a question and returns a response using RAG.
    """
    try:
        # Initialize documents if not already done
        await rag_service.initialize_docs()

        # Get the answer using RAG
        result = await rag_service.get_answer(request.text, strict_mode=request.strict_mode)

        return {"response": result["answer"]}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/ask")
async def ask(request: QuestionRequest) -> Dict[str, str]:
    """
    Ask endpoint that accepts a question and returns a response using RAG.
    """
    try:
        # Initialize documents if not already done
        await rag_service.initialize_docs()

        # Get the answer using RAG
        result = await rag_service.get_answer(request.text, strict_mode=request.strict_mode)

        return {"response": result["answer"]}
    except Exception as e:
        logger.error(f"Error processing ask request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Robotics AI Backend is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)