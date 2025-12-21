from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal

# Create a local Qdrant client
client = QdrantClient(':memory:')

# Check the actual type
print(f"Client type: {type(client)}")
print(f"Client _client type: {type(client._client)}")

# Check if _client has search
if hasattr(client._client, 'search'):
    print("client._client has search method")
else:
    print("client._client does NOT have search method")

# Check methods of the inner client
inner_methods = [m for m in dir(client._client) if not m.startswith('_') and callable(getattr(client._client, m))]
print(f"Inner client methods: {inner_methods[:10]}")

# Check specifically for search in inner client
search_methods = [m for m in inner_methods if 'search' in m.lower()]
print(f"Search methods in inner client: {search_methods}")