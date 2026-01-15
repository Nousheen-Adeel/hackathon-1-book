import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.markdown_parser import MarkdownParser
from embeddings.openai_client import OpenAIEmbeddingClient
from vector_db.qdrant_client import QdrantVectorDB


def ingest_documents():
    """Main function to ingest documents into the vector database."""
    print("Starting document ingestion process...")

    # Initialize components
    parser = MarkdownParser(docs_path="../docs")  # Relative to rag_backend
    embedding_client = OpenAIEmbeddingClient()
    vector_db = QdrantVectorDB()

    # Parse all documents
    print("Parsing markdown documents...")
    chunks = parser.parse_all_documents()
    print(f"Found {len(chunks)} text chunks to process.")

    # Generate embeddings for all chunks
    print("Generating embeddings...")
    embeddings = []
    processed_texts = []  # Track the actual text chunks that were embedded
    processed_count = 0

    for i, chunk in enumerate(chunks):
        text = chunk['text']
        original_metadata = chunk['metadata']

        # Chunk if too large
        token_count = embedding_client.num_tokens_from_text(text)
        if token_count > embedding_client.max_tokens_per_chunk:
            text_chunks = embedding_client.chunk_text(text, embedding_client.max_tokens_per_chunk)
            for j, subchunk_text in enumerate(text_chunks):
                embedding = embedding_client.get_embedding(subchunk_text)
                if embedding is not None:  # Check for None instead of truthiness
                    embeddings.append(embedding)
                    # Create new metadata for the subchunk, preserving original info
                    subchunk_metadata = original_metadata.copy()
                    subchunk_metadata['original_chunk_index'] = i
                    subchunk_metadata['subchunk_index'] = j
                    processed_texts.append({
                        'text': subchunk_text,
                        'metadata': subchunk_metadata
                    })
                    processed_count += 1
                    print(f"Processed embedding {processed_count}/{len(text_chunks)} for chunk {i+1}")
        else:
            embedding = embedding_client.get_embedding(text)
            if embedding is not None:  # Check for None instead of truthiness
                embeddings.append(embedding)
                processed_texts.append({
                    'text': text,
                    'metadata': original_metadata
                })
                processed_count += 1
                print(f"Processed embedding {processed_count}/{len(chunks)}")

    print(f"Generated {len(embeddings)} embeddings from {len(processed_texts)} processed text chunks.")

    # Store embeddings in Qdrant
    print("Storing embeddings in vector database...")
    vector_db.store_embeddings(processed_texts, embeddings)

    print("Document ingestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()