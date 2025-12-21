import uvicorn
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    # Import after loading environment
    from main import app
    print("Application imported successfully")
    print(f"Starting server on http://127.0.0.1:8000")
    
    # Start the server
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
    
except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)