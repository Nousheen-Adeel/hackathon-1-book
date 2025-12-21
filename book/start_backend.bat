@echo off
REM Script to start the RAG backend server

echo Starting RAG Chatbot Backend Server...
echo Make sure you have set up your environment variables in rag_backend/.env

REM Change to the rag_backend directory
cd rag_backend

REM Check if virtual environment exists, if not create one
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies if requirements.txt changed
echo Installing dependencies...
pip install -r requirements.txt

REM Start the FastAPI server
echo Starting FastAPI server on port 8000...
uvicorn main:app --reload --port 8000

pause