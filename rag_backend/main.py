from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict


class QuestionRequest(BaseModel):
    question: str


app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: QuestionRequest) -> Dict[str, str]:
    """
    Chat endpoint that accepts a question and returns a response.
    Currently returns a placeholder response.
    """
    # For now, return a simple placeholder response
    response = "Hello, I am your Robotics AI"
    return {"response": response}


@app.get("/")
async def root():
    return {"message": "Robotics AI Backend is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)