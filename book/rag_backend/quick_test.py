import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.markdown_parser import MarkdownParser
from embeddings.openrouter_client import OpenRouterEmbeddingClient
from vector_db.qdrant_client import QdrantVectorDB
import time

def quick_test_ingestion():
    """Quick test to verify the ingestion pipeline works."""
    print("Starting quick ingestion test...")
    
    # Initialize components
    print("1. Initializing components...")
    parser = MarkdownParser(docs_path="../docs")  # Relative to rag_backend
    embedding_client = OpenRouterEmbeddingClient()
    vector_db = QdrantVectorDB()
    
    # Parse just a single small document for testing
    print("2. Parsing a sample document...")
    from pathlib import Path
    sample_file = Path("../docs/intro.md")  # Use Path object
    if sample_file.exists():
        chunks = parser.parse_document(sample_file)
        print(f"Parsed {len(chunks)} chunks from sample document")
        
        if chunks:
            # Test embedding generation for first chunk only
            print("3. Testing embedding generation...")
            sample_text = chunks[0]['text'][:500]  # Use first 500 chars for test
            print(f"Sample text (first 100 chars): {sample_text[:100]}...")
            
            start_time = time.time()
            embedding = embedding_client.get_embedding(sample_text)
            embedding_time = time.time() - start_time
            
            if embedding:
                print(f"✓ Embedding generated successfully! Length: {len(embedding)}, Time: {embedding_time:.2f}s")
                
                # Test storing in vector DB
                print("4. Testing storage in vector database...")
                try:
                    vector_db.store_embeddings([chunks[0]], [embedding])
                    print("✓ Successfully stored in vector database!")
                    
                    # Test retrieval
                    print("5. Testing retrieval...")
                    search_results = vector_db.search_similar(embedding, top_k=1)
                    print(f"✓ Retrieved {len(search_results)} results from vector database")
                    
                    print("\n✓ All tests passed! The ingestion pipeline is working.")
                    return True
                except Exception as e:
                    print(f"✗ Storage/retrieval test failed: {e}")
                    return False
            else:
                print("✗ Failed to generate embedding")
                return False
        else:
            print("✗ No chunks found in sample document")
            return False
    else:
        print(f"✗ Sample file {sample_file} does not exist")
        # Try to find any markdown file
        for root, dirs, files in os.walk("../docs"):
            for file in files:
                if file.endswith(".md"):
                    sample_file = Path(os.path.join(root, file))
                    print(f"Found sample file: {sample_file}")
                    chunks = parser.parse_document(sample_file)
                    if chunks:
                        print(f"Test with first chunk of {file}")
                        sample_text = chunks[0]['text'][:500]
                        embedding = embedding_client.get_embedding(sample_text)
                        if embedding:
                            print("✓ Embedding works with actual document!")
                            return True
                    break
        return False

if __name__ == "__main__":
    quick_test_ingestion()