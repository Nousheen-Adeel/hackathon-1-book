import os
import logging
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings using OpenAI API"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        # Debug info for terminal
        print(f"--- EMBEDDING DEBUG ---")
        print(f"Model: {self.model}")
        print(f"API Key: {self.api_key[:10]}...")
        print(f"-----------------------")

        self.client = OpenAI(
            api_key=self.api_key
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Sanitize input texts to remove problematic characters
        sanitized_texts = []
        for text in texts:
            # Remove null characters and other problematic characters for OpenAI API
            sanitized = text.replace('\x00', '')  # Remove null characters
            sanitized = sanitized.replace('\ud83d', '')  # Remove surrogate characters
            sanitized = sanitized.strip()
            # Ensure text is not empty after sanitization
            if sanitized:
                sanitized_texts.append(sanitized)
            else:
                # If sanitized text is empty, add a placeholder to maintain alignment
                sanitized_texts.append(" ")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=sanitized_texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"OpenAI Error: {e}")

    def get_embedding(self, text: str) -> List[float]:
        # Sanitize input text to remove problematic characters
        sanitized_text = text.replace('\x00', '').strip()
        return self.generate_embeddings([sanitized_text])[0]