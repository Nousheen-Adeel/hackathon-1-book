import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.markdown_parser import MarkdownParser
from embeddings.openrouter_client import OpenRouterEmbeddingClient
from vector_db.qdrant_client import QdrantVectorDB


def ingest_documents():
    """Main function to ingest documents into the vector database."""
    print("Starting document ingestion process...")
    
    # Initialize components
    parser = MarkdownParser(docs_path="../docs")  # Relative to rag_backend
    embedding_client = OpenRouterEmbeddingClient()
    vector_db = QdrantVectorDB()
    
    # Parse all documents
    print("Parsing markdown documents...")
    chunks = parser.parse_all_documents()
    print(f"Found {len(chunks)} text chunks to process.")
    
    # Generate embeddings for all chunks
    print("Generating embeddings...")
    embeddings = []
    processed_count = 0
    
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        
        # Chunk if too large
        token_count = embedding_client.num_tokens_from_text(text)
        if token_count > embedding_client.max_tokens_per_chunk:
            text_chunks = embedding_client.chunk_text(text, embedding_client.max_tokens_per_chunk)
            for subchunk_text in text_chunks:
                embedding = embedding_client.get_embedding(subchunk_text)
                if embedding:
                    embeddings.append(embedding)
                    processed_count += 1
                    print(f"Processed embedding {processed_count}/{len(text_chunks)} for chunk {i+1}")
        else:
            embedding = embedding_client.get_embedding(text)
            if embedding:
                embeddings.append(embedding)
                processed_count += 1
                print(f"Processed embedding {processed_count}/{len(chunks)}")
    
    print(f"Generated {len(embeddings)} embeddings from {len(chunks)} text chunks.")
    
    # Store embeddings in Qdrant
    print("Storing embeddings in vector database...")
    vector_db.store_embeddings(chunks, embeddings)
    
    print("Document ingestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()