# # rag_backend\api\rag_service.py
# from fastapi import HTTPException
# import logging
# from typing import List, Dict
# import asyncio
# import os
# from dotenv import load_dotenv

# from data_ingestion.loader import DocumentLoader
# from embeddings.generator import EmbeddingGenerator
# from vector_db.qdrant import VectorDB

# load_dotenv()

# logger = logging.getLogger(__name__)

# class RAGService:
#     """Main service class for RAG functionality"""
    
#     def __init__(self):
#         try:
#             self.embedding_generator = EmbeddingGenerator()
#         except Exception as e:
#             logger.warning(f"Could not initialize embedding generator: {e}")
#             # Create a mock embedding generator that returns dummy embeddings
#             class MockEmbeddingGenerator:
#                 def generate_embeddings(self, texts):
#                     import numpy as np
#                     return [[0.01] * 1536 for _ in texts]

#                 def get_embedding(self, text):
#                     import numpy as np
#                     return [0.01] * 1536

#             self.embedding_generator = MockEmbeddingGenerator()

#         try:
#             self.vector_db = VectorDB()
#         except Exception as e:
#             logger.warning(f"Could not initialize vector database: {e}")
#             # Create a mock vector database
#             class MockVectorDB:
#                 def search_similar(self, query_embedding, top_k=5):
#                     return []

#                 def store_embeddings(self, chunks, embeddings):
#                     pass

#             self.vector_db = MockVectorDB()

#         self.docs_loaded = False
#         self.initialization_error = None

#     async def initialize_docs(self):
#         """Load and index documents if not already loaded"""
#         if self.docs_loaded:
#             return

#         if self.initialization_error:
#             # Reset the error flag to allow retry if conditions have changed (e.g., new API key)
#             logger.info(f"Retrying document initialization after previous error: {self.initialization_error}")
#             self.initialization_error = None

#         try:
#             # Load documents
#             loader = DocumentLoader()
#             documents = loader.load_documents()

#             if not documents:
#                 logger.warning("No documents found to load")
#                 # Still mark as initialized so the system can work with LLM directly
#                 self.docs_loaded = True
#                 return

#             # Chunk documents
#             chunks = loader.chunk_documents(documents)

#             if not chunks:
#                 logger.warning("No chunks created from documents")
#                 # Still mark as initialized so the system can work with LLM directly
#                 self.docs_loaded = True
#                 return

#             # Generate embeddings for all chunks
#             texts = [chunk["content"] for chunk in chunks]
#             embeddings = self.embedding_generator.generate_embeddings(texts)

#             # Store in vector database
#             self.vector_db.store_embeddings(chunks, embeddings)
#             self.docs_loaded = True

#             logger.info(f"Successfully initialized {len(chunks)} document chunks in vector database")
#         except Exception as e:
#             self.initialization_error = str(e)
#             logger.error(f"Error initializing documents: {str(e)}")
#             # Still mark as loaded so the system can work with LLM directly
#             self.docs_loaded = True
#             # Don't raise the exception during startup, just log it
#             # The system can still work if the LLM is available but documents are not indexed yet

#     def get_collection_info(self):
#         """Get information about the vector database collection for debugging"""
#         try:
#             return self.vector_db.get_collection_info()
#         except Exception as e:
#             logger.error(f"Error getting collection info: {str(e)}")
#             return None
    
#     async def get_answer(self, query: str, top_k: int = 5, strict_mode: bool = False) -> Dict:
#         """
#         Get answer to query using RAG (Retrieval Augmented Generation)
#         """
#         try:
#             # Generate embedding for the query
#             query_embedding = self.embedding_generator.get_embedding(query)

#             # Search for similar documents in the vector database
#             similar_docs = self.vector_db.search_similar(query_embedding, top_k=top_k)

#             if similar_docs:
#                 # Format the context from retrieved documents
#                 context_parts = []
#                 for doc in similar_docs:
#                     context_parts.append(f"Source: {doc['source']}\nContent: {doc['content']}")

#                 context = "\n\n".join(context_parts)
#             else:
#                 context = "No relevant documents found in the knowledge base."
#         except Exception as e:
#             logger.warning(f"Error during document retrieval: {str(e)}. Using empty context.")
#             similar_docs = []
#             context = "No relevant documents found in the knowledge base."

#         # Generate answer using the context and query
#         answer = await self._generate_answer_with_llm(query, context, strict_mode)

#         return {
#             "answer": answer,
#             "sources": similar_docs,
#             "query": query
#         }
    
#         async def _generate_answer_with_llm(self, query: str, context: str, strict_mode: bool = False) -> str:
#             """
#             Generate answer using direct OpenAI API
#             """
#             from openai import AsyncOpenAI

#             api_key = os.getenv("OPENAI_API_KEY")
#             model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

#             if not api_key:
#                 return "API key not configured. Please set OPENAI_API_KEY in environment variables."

#             client = AsyncOpenAI(api_key=api_key)

#             if strict_mode:
#                 system_prompt = "You are an expert assistant for the textbook 'Physical AI & Humanoid Robotics'. Answer ONLY based on the provided context. If the answer is not in the context, say 'Answer not found in the book.'"
#             else:
#                 system_prompt = "You are an expert assistant for the textbook 'Physical AI & Humanoid Robotics'. Use the provided context to answer accurately. If context is irrelevant, use your knowledge."

#             user_prompt = f"""Context:
#     {context}

#     Question: {query}

#     Answer:"""

#             try:
#                 response = await client.chat.completions.create(
#                     model=model,
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_prompt}
#                     ],
#                     max_tokens=1000,
#                     temperature=0.3
#                 )
#                 return response.choices[0].message.content.strip()
#             except Exception as e:
#                 logger.error(f"Error calling OpenAI API: {str(e)}")
#                 if "not found in the book" in context.lower():
#                     return "Answer not found in the book."
#                 return f"Sorry, I couldn't process your question due to a service error."

#     def cleanup_project_files(self):
#         """Call the file cleanup utility"""
#         from ..utils.file_cleanup import cleanup_project_files
#         cleanup_project_files()










# rag_backend/api/rag_service.py

from fastapi import HTTPException
import logging
from typing import List, Dict
import asyncio
import os
from dotenv import load_dotenv
from data_ingestion.loader import DocumentLoader
from embeddings.generator import EmbeddingGenerator
from vector_db.qdrant import VectorDB

load_dotenv()

logger = logging.getLogger(__name__)

class RAGService:
    """Main service class for RAG functionality"""

    def __init__(self):
        try:
            self.embedding_generator = EmbeddingGenerator()
        except Exception as e:
            logger.warning(f"Could not initialize embedding generator: {e}")
            # Create a mock embedding generator that returns dummy embeddings
            class MockEmbeddingGenerator:
                def generate_embeddings(self, texts):
                    import numpy as np
                    vector_size = int(os.getenv("VECTOR_SIZE", 1536))
                    return [[0.01] * vector_size for _ in texts]

                def get_embedding(self, text):
                    import numpy as np
                    vector_size = int(os.getenv("VECTOR_SIZE", 1536))
                    return [0.01] * vector_size

            self.embedding_generator = MockEmbeddingGenerator()

        try:
            self.vector_db = VectorDB()
        except Exception as e:
            logger.warning(f"Could not initialize vector database: {e}")
            # Create a mock vector database
            class MockVectorDB:
                def search_similar(self, query_embedding, top_k=5):
                    return []

                def store_embeddings(self, chunks, embeddings):
                    pass

                def get_collection_info(self):
                    return None

            self.vector_db = MockVectorDB()

        self.docs_loaded = False
        self.initialization_error = None

    async def initialize_docs(self):
        """Load and index documents if not already loaded"""
        if self.docs_loaded:
            return

        if self.initialization_error:
            logger.info(f"Retrying document initialization after previous error: {self.initialization_error}")
            self.initialization_error = None

        try:
            # Load documents
            loader = DocumentLoader()
            documents = loader.load_documents()

            if not documents:
                logger.warning("No documents found to load")
                self.docs_loaded = True
                return

            # Chunk documents
            chunks = loader.chunk_documents(documents)

            if not chunks:
                logger.warning("No chunks created from documents")
                self.docs_loaded = True
                return

            # Generate embeddings for all chunks
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)

            # Store in vector database
            self.vector_db.store_embeddings(chunks, embeddings)

            self.docs_loaded = True
            logger.info(f"Successfully initialized {len(chunks)} document chunks in vector database")

        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Error initializing documents: {str(e)}")
            self.docs_loaded = True  # Allow LLM fallback

    def get_collection_info(self):
        """Get information about the vector database collection for debugging"""
        try:
            return self.vector_db.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return None

    async def get_answer(self, query: str, top_k: int = 5, strict_mode: bool = False) -> Dict:
        """
        Get answer to query using RAG (Retrieval Augmented Generation)
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.get_embedding(query)

            # Search for similar documents
            similar_docs = self.vector_db.search_similar(query_embedding, top_k=top_k)

            if similar_docs:
                context_parts = []
                for doc in similar_docs:
                    context_parts.append(f"Source: {doc['source']}\nContent: {doc['content']}")
                context = "\n\n".join(context_parts)
            else:
                context = "No relevant documents found in the knowledge base."

        except Exception as e:
            logger.warning(f"Error during document retrieval: {str(e)}. Using empty context.")
            similar_docs = []
            context = "No relevant documents found in the knowledge base."

        # Generate answer using LLM
        answer = await self._generate_answer_with_llm(query, context, strict_mode)

        return {
            "answer": answer,
            "sources": similar_docs,
            "query": query
        }

    async def _generate_answer_with_llm(self, query: str, context: str, strict_mode: bool = False) -> str:
        """
        Generate answer using direct OpenAI API
        """
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        if not api_key:
            return "API key not configured. Please set OPENAI_API_KEY in environment variables."

        client = AsyncOpenAI(api_key=api_key)

        if strict_mode:
            system_prompt = (
                "You are an expert assistant for the textbook 'Physical AI & Humanoid Robotics'. "
                "Answer ONLY based on the provided context. "
                "If the answer is not in the context, respond with 'Answer not found in the book.'"
            )
        else:
            system_prompt = (
                "You are an expert assistant for the textbook 'Physical AI & Humanoid Robotics'. "
                "Use the provided context to answer accurately. "
                "If the context is irrelevant or insufficient, use your general knowledge to provide a helpful response."
            )

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            if "not found" in context.lower():
                return "Answer not found in the book."
            return "Sorry, I couldn't process your question due to a service error."

    def cleanup_project_files(self):
        """Call the file cleanup utility"""
        try:
            from ..utils.file_cleanup import cleanup_project_files
            cleanup_project_files()
        except Exception as e:
            logger.warning(f"Could not run cleanup: {e}")