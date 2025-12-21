import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

print("Testing RAGService initialization...")

try:
    from api.rag_service import RAGService
    print("Import successful")
    
    print("Creating RAGService instance...")
    rag_service = RAGService()
    print("RAGService created successfully")
    
    print("Testing get_answer method...")
    import asyncio
    
    async def test():
        try:
            result = await rag_service.get_answer('test query')
            print(f"get_answer result: {result}")
        except Exception as e:
            print(f"Error in get_answer: {e}")
            traceback.print_exc()
    
    asyncio.run(test())
    print("Test completed successfully")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    traceback.print_exc()