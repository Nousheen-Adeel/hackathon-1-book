from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid

# Create a local Qdrant client
client = QdrantClient(
    url="https://499fc875-1fc3-46b9-8d0c-ebc329798849.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1gabqxivh6-T706S8O8YrrO6haY1nm1crNdD-YUl1SU"
)


# Create a collection
collection_name ="book_docs"
vector_size = 1536

try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("Collection created successfully")
except Exception as e:
    print(f"Error creating collection: {e}")

# Add a test point
test_embedding = [0.1] * vector_size  # Simple test embedding
test_point = PointStruct(
    id=str(uuid.uuid4()),
    vector=test_embedding,
    payload={
        "content": "Test content",
        "source": "test_source",
        "title": "test_title",
        "chunk_id": "test_chunk"
    }
)

try:
    client.upsert(
        collection_name=collection_name,
        points=[test_point]
    )
    print("Point added successfully")
except Exception as e:
    print(f"Error adding point: {e}")

# Try to search
try:
    results = client.search(
        collection_name=collection_name,
        query_vector=test_embedding,
        limit=5
    )
    print(f"Search successful, found {len(results)} results")
except Exception as e:
    print(f"Error during search: {e}")
    print("Available methods on client object:")
    # Try to get methods in a safer way
    safe_attrs = []
    for attr in dir(client):
        if not attr.startswith('_'):
            try:
                obj = getattr(client, attr)
                if callable(obj):
                    safe_attrs.append(attr)
            except:
                continue
    print(safe_attrs[:20])  # First 20 attributes