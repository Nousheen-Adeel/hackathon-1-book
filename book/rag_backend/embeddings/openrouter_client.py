import os
import openai
from dotenv import load_dotenv
import tiktoken


class OpenRouterEmbeddingClient:
    """Client for generating embeddings using OpenRouter API."""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        # Configure OpenAI client to use OpenRouter
        openai.base_url = "https://openrouter.ai/api/v1/"
        openai.api_key = self.api_key

        # Use a proper embedding model instead of a chat model
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")  # OpenRouter format
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))

        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Close approximation
    
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a text using OpenRouter API."""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return []
    
    def num_tokens_from_text(self, text: str) -> int:
        """Return the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = None) -> list[str]:
        """
        Split text into chunks that are no larger than max_tokens.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens_per_chunk
            
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Split into chunks of max_tokens
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks