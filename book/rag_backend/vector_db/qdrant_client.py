import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from dotenv import load_dotenv


class QdrantVectorDB:
    """Integration with Qdrant vector database for storing and searching embeddings."""
    
    def __init__(self):
        load_dotenv()
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings")

        # Initialize Qdrant client - use local mode if no URL provided or URL is empty/default
        if self.url and self.url != "your_qdrant_url_here":
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                prefer_grpc=False  # Using REST API for simplicity
            )
        else:
            # Use local/disk mode with a local path
            from pathlib import Path
            local_path = Path("./qdrant_data").resolve()
            local_path.mkdir(exist_ok=True)
            self.client = QdrantClient(path=str(local_path))  # Local mode
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create the collection if it doesn't exist already."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [coll.name for coll in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection - dimension is based on embedding size (typically 384 for smaller models, 1536 for larger)
                # Using 1536 as it's common for embedding models - adjust based on actual embedding size
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                
                # Create payload index for metadata searching
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="file_path",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                print(f"Created collection '{self.collection_name}' successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise e
    
    def store_embeddings(self, texts: List[Dict], embeddings: List[List[float]]):
        """
        Store embeddings with their associated text and metadata.
        
        Args:
            texts: List of dictionaries containing 'text' and 'metadata'
            embeddings: List of embedding vectors
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        points = []
        for i, (text_obj, embedding) in enumerate(zip(texts, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text_obj['text'],
                    "metadata": text_obj['metadata']
                }
            )
            points.append(point)
        
        # Upload points to Qdrant
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Stored {len(points)} embeddings in Qdrant collection '{self.collection_name}'.")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar embeddings in the vector database.
        
        Args:
            query_embedding: The embedding to search for similarities
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar text snippets and metadata
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        results = []
        for hit in search_result:
            result = {
                'text': hit.payload['text'],
                'metadata': hit.payload['metadata'],
                'score': hit.score
            }
            results.append(result)
        
        return results
    
    def search_by_metadata(self, metadata_filter: Dict, top_k: int = 5) -> List[Dict]:
        """
        Search for embeddings with specific metadata filters.
        """
        from qdrant_client.http.models import FieldCondition, MatchValue
        
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                )
            )
        
        if conditions:
            # Create a filter combining all conditions with AND logic
            from qdrant_client.http.models import Filter
            search_filter = Filter(must=conditions)
        else:
            search_filter = None
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_filter=search_filter,
            limit=top_k
        )
        
        results = []
        for hit in search_result:
            result = {
                'text': hit.payload['text'],
                'metadata': hit.payload['metadata'],
                'score': hit.score
            }
            results.append(result)
        
        return results