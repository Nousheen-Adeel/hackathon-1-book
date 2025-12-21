import os
import sys

# Set environment to use local Qdrant before importing anything
os.environ['QDRANT_URL'] = ''  # Empty to trigger local mode

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the FastAPI app
if __name__ == "__main__":
    print("Starting the RAG API server...")
    print("Using local Qdrant mode")
    
    from main import app
    import uvicorn
    
    print("Server ready! Access at http://127.0.0.1:8000")
    print("Docs available at http://127.0.0.1:8000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)