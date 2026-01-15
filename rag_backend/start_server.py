import uvicorn
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    # Import after loading environment - use simple_server which has full RAG implementation
    from simple_server import app

    # Get port from environment variable, default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")  # Use 0.0.0.0 for Railway deployment

    print("Application imported successfully")
    print(f"Starting server on http://{host}:{port}")

    # Start the server with dynamic port
    uvicorn.run(app, host=host, port=port, log_level='info')

except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)