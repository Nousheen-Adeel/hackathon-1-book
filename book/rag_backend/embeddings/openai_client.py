import os
import openai
from dotenv import load_dotenv
import tiktoken


class OpenAIEmbeddingClient:
    """Client for generating embeddings using OpenAI API directly."""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Configure OpenAI client to use OpenAI directly
        openai.api_key = self.api_key

        # Use the embedding model specified in environment
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))

        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Close approximation

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a text using OpenAI API."""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return None to indicate failure, rather than an empty list
            # This way the calling code can skip invalid embeddings
            return None

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