# Terminal Commands to Start Both Services

## Starting the FastAPI Backend (Port 8000)

Open a terminal/command prompt and navigate to the rag_backend directory:

```bash
cd C:\Users\Dell\ai-book\rag_backend
```

Then run the following command to start the FastAPI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Alternatively, if you have the script set up to run directly:

```bash
python main.py
```

## Starting the Docusaurus Frontend (Port 3000)

Open another terminal/command prompt and navigate to the rag_frontend directory:

```bash
cd C:\Users\Dell\ai-book\rag_frontend
```

Then run the following command to start the Docusaurus development server:

```bash
npm run start
```

## Important Notes

1. Make sure you have the required dependencies installed:
   - For FastAPI backend: `pip install fastapi uvicorn python-multipart`
   - For Docusaurus frontend: `npm install` (if you haven't already)

2. The backend must be running before using the chatbot on the frontend.

3. Once both services are running, you can access:
   - Docusaurus frontend at: http://localhost:3000
   - FastAPI backend at: http://127.0.0.1:8000
   - Chat endpoint at: http://127.0.0.1:8000/chat