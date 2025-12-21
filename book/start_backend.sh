#!/bin/bash
# Script to start the RAG backend server

echo "Starting RAG Chatbot Backend Server..."
echo "Make sure you have set up your environment variables in rag_backend/.env"

# Change to the rag_backend directory
cd rag_backend

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if requirements.txt changed
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the FastAPI server
echo "Starting FastAPI server on port 8000..."
uvicorn main:app --reload --port 8000