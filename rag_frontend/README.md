# Physical AI & Humanoid Robotics - RAG Frontend

Simple web interface for interacting with the RAG chatbot system.

## How to Use

1. Make sure the backend server is running on `http://localhost:8000`
2. Open `index.html` in your web browser
3. Type your question about the textbook in the input field
4. Click "Send Question" or press Enter to get an answer

## Features

- Clean, responsive chat interface
- Real-time display of questions and answers
- Source references for retrieved documents
- Error handling for API communication issues
- Loading indicators during processing

## Configuration

The frontend communicates with the backend at `http://localhost:8000/chat` by default.
If you need to change this, update the fetch URL in the JavaScript code.