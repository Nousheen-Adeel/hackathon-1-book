import os
from typing import List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load documents from the book/docs/ directory"""
    
    def __init__(self, docs_path: str = "../../book/docs"):
        self.docs_path = Path(docs_path)
        
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all markdown documents from the docs directory
        Returns a list of dictionaries with 'content' and 'source' keys
        """
        documents = []
        
        if not self.docs_path.exists():
            logger.error(f"Docs path does not exist: {self.docs_path}")
            return documents
            
        # Look for markdown files in the docs directory and subdirectories
        for md_file in self.docs_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Create document object
                doc = {
                    "content": content,
                    "source": str(md_file.relative_to(self.docs_path)),
                    "title": md_file.stem  # filename without extension
                }
                
                documents.append(doc)
                logger.info(f"Loaded document: {doc['source']}")
                
            except Exception as e:
                logger.error(f"Error loading document {md_file}: {str(e)}")
                
        logger.info(f"Loaded {len(documents)} documents from {self.docs_path}")
        return documents

    def chunk_documents(self, documents: List[Dict], chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
        """
        Split documents into chunks for better retrieval
        """
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            source = doc["source"]
            title = doc["title"]
            
            # Simple sliding window approach to chunk the content
            start = 0
            while start < len(content):
                end = start + chunk_size
                
                # If we're near the end, make sure to include the remainder
                if end > len(content):
                    end = len(content)
                    
                chunk_text = content[start:end]
                
                chunk = {
                    "content": chunk_text,
                    "source": source,
                    "title": title,
                    "chunk_id": f"{source}_{start}_{end}"
                }
                
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start += chunk_size - overlap
                
                # If start is beyond the length, break
                if start >= len(content):
                    break
                    
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks