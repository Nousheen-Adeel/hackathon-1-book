import os
# Make sure we're using local mode
os.environ['QDRANT_URL'] = ''  # Empty string to trigger local mode

from vector_db.qdrant_client import QdrantVectorDB

try:
    print("Creating QdrantVectorDB instance...")
    db = QdrantVectorDB()
    print("SUCCESS: Qdrant client created in local mode")
    print(f"Collection: {db.collection_name}")
    
    # Test if we can count items (should be 0 initially)
    count = db.client.count(db.collection_name).count if db.client.collection_exists(db.collection_name) else 0
    print(f"Current items in collection: {count}")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()