import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from simple_server import app
import uvicorn

print('Starting server on http://127.0.0.1:8000')
print('Server is running. Press Ctrl+C to stop.')
try:
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
except KeyboardInterrupt:
    print("Server stopped by user")
except Exception as e:
    print(f"Error running server: {e}")
    import traceback
    traceback.print_exc()