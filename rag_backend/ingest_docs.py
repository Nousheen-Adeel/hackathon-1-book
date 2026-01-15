# book/rag_backend/ingest_docs.py

import requests
import uuid
from bs4 import BeautifulSoup
import trafilatura

from vector_db.qdrant import VectorDB
from embeddings.generator import EmbeddingGenerator

SITEMAP_URL = "https://physicalhumanoidaitextbook.vercel.app/sitemap.xml"


def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def get_urls_from_sitemap():
    resp = requests.get(SITEMAP_URL, timeout=20)
    soup = BeautifulSoup(resp.text, "xml")
    return [loc.text for loc in soup.find_all("loc")]


def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded)


def ingest():
    vector_db = VectorDB()
    embedder = EmbeddingGenerator()

    urls = get_urls_from_sitemap()
    print(f"Found {len(urls)} URLs in sitemap")

    chunks_to_store = []
    embeddings = []

    for url in urls:
        text = extract_text_from_url(url)
        if not text or len(text) < 200:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = embedder.get_embedding(chunk)
            if embedding:
                chunks_to_store.append({
                    "content": chunk,
                    "source": url,
                    "title": url.split("/")[-1],
                    "chunk_id": f"{url}_{i}"
                })
                embeddings.append(embedding)

    print(f"Storing {len(chunks_to_store)} chunks in Qdrant")
    vector_db.store_embeddings(chunks_to_store, embeddings)


if __name__ == "__main__":
    ingest()
