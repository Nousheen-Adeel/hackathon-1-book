import uuid
import os
import logging
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url == "local":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_docs")
        
        # Sahi vector size select karna
        embedding_model = os.getenv("EMBEDDING_MODEL", "").lower()
        if "nomic" in embedding_model:
            self.vector_size = 768
        elif "text-embedding-3-small" in embedding_model:
            self.vector_size = 1536
        else:
            self.vector_size = int(os.getenv("VECTOR_SIZE", "1536"))

        self.collection_initialized = False

    def _create_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            self.collection_initialized = True
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection {self.collection_name} (Size: {self.vector_size})")
            self.collection_initialized = True

    def store_embeddings(self, chunks: List[Dict], embeddings: List[List[float]]):
        if not self.collection_initialized: self._create_collection()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={**chunk, "chunk_id": chunk.get("chunk_id", f"idx_{i}")}
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Stored {len(points)} points.")

    def search_similar(self, query_embedding: List[float], top_k: int = 5):
        """Search for similar embeddings in the vector database."""
        if not self.collection_initialized:
            self._create_collection()

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        results = []
        for hit in search_results:
            result = {
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score
            }
            results.append(result)

        return results

    def get_collection_info(self):
        """Get information about the collection for debugging."""
        if not self.collection_initialized:
            self._create_collection()

        try:
            collection_info = self.client.get_collection(self.collection_name)
            # Just return basic info that we know is available
            return {
                "name": self.collection_name,
                "vector_size": self.vector_size,  # Use the internally stored vector size
                "point_count": collection_info.point_count
            }
        except Exception:
            # Collection might not exist yet
            return None