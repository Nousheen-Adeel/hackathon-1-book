from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
import openai
import os
from dotenv import load_dotenv
from embeddings.openrouter_client import OpenRouterEmbeddingClient
from vector_db.qdrant_client import QdrantVectorDB


# Load environment variables
load_dotenv()

# Configure OpenAI client to use OpenRouter
openai.base_url = "https://openrouter.ai/api/v1/"
openai.api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize global components for RAG
embedding_client = OpenRouterEmbeddingClient()
vector_db = QdrantVectorDB()

if not openai.api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    temperature: Optional[float] = 0.7


class SelectedTextChatRequest(ChatRequest):
    selected_text: str


app = FastAPI(title="Book RAG Chatbot API",
              description="Retrieval-Augmented Generation chatbot for the Physical AI & Humanoid Robotics book",
              version="1.0.0")


def format_context_for_llm(retrieved_chunks: List[Dict]) -> str:
    """Format retrieved document chunks for LLM context."""
    formatted_context = ""
    sources = []
    
    for i, chunk in enumerate(retrieved_chunks):
        text = chunk['text']
        metadata = chunk['metadata']
        
        formatted_context += f"\n--- Context Chunk {i+1} ---\n{text}\n"
        
        # Track sources
        source_info = {
            'title': metadata.get('title', 'Unknown'),
            'file_path': metadata.get('file_path', 'Unknown'),
            'heading': metadata.get('heading', ''),
            'score': chunk.get('score', 0.0)
        }
        sources.append(source_info)
    
    return formatted_context, sources


@app.get("/")
async def root():
    return {"message": "Book RAG Chatbot API is running!"}


@app.post("/chat")
async def chat_full_book(request: ChatRequest):
    """
    Chat endpoint that uses full book content for RAG.
    """
    try:
        # Generate embedding for the user query
        query_embedding = embedding_client.get_embedding(request.message)
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for query")
        
        # Search for relevant chunks in the vector database
        retrieved_chunks = vector_db.search_similar(query_embedding, top_k=5)
        
        if not retrieved_chunks:
            # If no relevant chunks found, respond accordingly
            return {
                "response": "I couldn't find relevant information in the book to answer your question.",
                "sources": []
            }
        
        # Format retrieved context for the LLM
        context, sources = format_context_for_llm(retrieved_chunks)
        
        # Construct the system message with context
        system_message = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics book. 
        Use the following context from the book to answer the user's question.
        Be accurate, detailed, and cite sources when possible based on the context provided.
        
        Context from the book:
        {context}
        """
        
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        # Add history if available
        for msg in request.history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add user message
        messages.append({"role": "user", "content": request.message})

        # Call OpenRouter API
        response = openai.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "qwen/qwen-2-7b-instruct:free"),
            messages=messages,
            temperature=request.temperature,
            max_tokens=1024,
        )

        return {
            "response": response.choices[0].message.content,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/selected")
async def chat_selected_text(request: SelectedTextChatRequest):
    """
    Chat endpoint that uses only user-selected text for context.
    """
    try:
        # Validate that selected_text exists
        if not request.selected_text.strip():
            raise HTTPException(status_code=400, detail="Selected text cannot be empty")
        
        # Construct the prompt with the selected text as context
        system_message = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics book. 
        Answer only based on the following selected text from the book.
        If the question cannot be answered from the provided text, clearly state that the information is not available in the selected text.
        
        Selected text: {request.selected_text}
        """
        
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        # Add history if available
        for msg in request.history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add user message
        messages.append({"role": "user", "content": request.message})

        # Call OpenRouter API
        response = openai.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "qwen/qwen-2-7b-instruct:free"),
            messages=messages,
            temperature=request.temperature,
            max_tokens=1024,
        )

        # Check if response indicates information is not in selected text
        response_text = response.choices[0].message.content
        if "information is not available" in response_text.lower() or "cannot answer" in response_text.lower():
            return {
                "response": response_text,
                "sources": [],
                "warning": "The requested information may not be available in the selected text."
            }
        
        return {
            "response": response_text,
            "sources": [{"text": request.selected_text[:200] + "..."}]  # Truncated for brevity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}