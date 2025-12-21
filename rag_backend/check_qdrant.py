from qdrant_client import QdrantClient
import inspect

# Create a local Qdrant client
client = QdrantClient(':memory:')

# Print all methods
all_attrs = dir(client)
methods = []
for attr in all_attrs:
    if not attr.startswith('_'):
        obj = getattr(client, attr)
        if callable(obj):
            methods.append(attr)

print("Available methods:", methods)

# Check specifically for search-related methods
search_methods = [m for m in methods if 'search' in m.lower()]
print("Search-related methods:", search_methods)

# Check if 'search' method exists
has_search = hasattr(client, 'search')
print(f"Has 'search' method: {has_search}")