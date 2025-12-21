# Physical AI & Humanoid Robotics Book

This is a Docusaurus-based textbook on Physical AI & Humanoid Robotics with integrated RAG chatbot functionality.

## Running the Application

### Prerequisites
- Node.js 16 or higher
- Python 3.11
- The RAG backend server running on `http://localhost:8000`

### Setup

1. **Start the RAG backend server**:
   ```bash
   cd ../rag_backend
   pip install -r requirements.txt
   python main.py
   ```

2. **Install and run the Docusaurus frontend**:
   ```bash
   npm install
   npm run start
   ```

### Development Notes

- The chatbot component is integrated into the layout and accessible via the floating button
- The chatbot communicates with the backend at `http://localhost:8000`
- For production deployment, ensure your hosting solution allows requests to the backend API

### Production Deployment

For production deployment, you'll need to:
1. Deploy the backend server to a public URL
2. Update the API_BASE in `src/components/ChatbotAPI.js` to point to your production backend
3. Ensure CORS settings allow requests from your frontend domain