import uuid
from typing import List, Dict, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http import models
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VectorDB:
    """Qdrant vector database wrapper for storing and retrieving document embeddings"""
    
    def __init__(self):
        # Configure Qdrant client
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Check if we're using local mode
        if qdrant_url == "local":
            # Use local in-memory mode
            self.client = QdrantClient(":memory:")
            self.qdrant_url = "local"
        else:
            # Initialize Qdrant client with URL
            if qdrant_api_key:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.client = QdrantClient(url=qdrant_url)
            self.qdrant_url = qdrant_url

        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_docs")
        self.vector_size = int(os.getenv("VECTOR_SIZE", "1536"))  # Default for OpenAI embeddings

        # Create collection if it doesn't exist - will be done when first needed
        self.collection_initialized = False
    
    def _create_collection(self):
        """Create the collection if it doesn't exist"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
            self.collection_initialized = True
        except:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{self.collection_name}'")
            self.collection_initialized = True

    def _ensure_collection_exists(self):
        """Ensure the collection exists before performing operations"""
        if not self.collection_initialized:
            self._create_collection()
    
    def store_embeddings(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Store document chunks with their embeddings in the vector database
        """
        try:
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {str(e)}")
            raise

        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}")
                }
            )
            points.append(point)

        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Stored {len(points)} embeddings in collection '{self.collection_name}'")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents based on the query embedding
        """
        try:
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {str(e)}")
            # Return empty list if Qdrant is not available
            return []

        # For local Qdrant client, we need to use the inner client directly
        if hasattr(self.client, '_client') and hasattr(self.client._client, 'search'):
            # Use the inner client for local mode
            results = self.client._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
        else:
            # Use the regular client for remote mode
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

        # Extract the payload from search results
        similar_docs = []
        for result in results:
            doc = {
                "content": result.payload["content"],
                "source": result.payload["source"],
                "title": result.payload["title"],
                "score": result.score
            }
            similar_docs.append(doc)

        logger.info(f"Found {len(similar_docs)} similar documents")
        return similar_docs
    
    def clear_collection(self):
        """
        Clear all points from the collection (useful for reindexing)
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Cleared collection '{self.collection_name}'")
            self._create_collection()  # Recreate empty collection
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")