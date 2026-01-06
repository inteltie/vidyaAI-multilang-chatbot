import asyncio
import os
import logging
from dotenv import load_dotenv
from services.retriever import RetrieverService
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_filters():
    load_dotenv()
    
    # Initialize retriever
    retriever = RetrieverService(settings)
    
    query = "Tell me about photosynthesis"
    
    print(f"\n--- Testing Query: '{query}' ---")
    
    # 1. No filters
    print("\n1. No filters:")
    docs = await retriever.retrieve(query_en=query, filters=None)
    print(f"   Found {len(docs)} documents.")
    if docs:
        for i, doc in enumerate(docs[:3], 1):
             meta = doc.get("metadata", {})
             print(f"      [{i}] Score: {doc.get('score'):.3f} | Subject: {meta.get('subject')} | Chapter: {meta.get('chapter')}")

    # 2. Subject filter
    subject = "Daily news" 
    print(f"\n2. Filter by subject='{subject}':")
    docs = await retriever.retrieve(query_en=query, filters={"subject": subject})
    print(f"   Found {len(docs)} documents.")
    for doc in docs[:3]:
        meta = doc.get("metadata", {})
        print(f"      Score: {doc.get('score'):.3f} | Subject: {meta.get('subject')}")

    # 3. Lecture ID filter (Integer)
    lecture_id = 210
    print(f"\n3. Filter by lecture_id={lecture_id} (Integer):")
    docs = await retriever.retrieve(query_en=query, filters={"lecture_id": lecture_id})
    print(f"   Found {len(docs)} documents.")
    for doc in docs[:3]:
        meta = doc.get("metadata", {})
        print(f"      Score: {doc.get('score'):.3f} | Lecture ID: {meta.get('lecture_id')}")

    # 4. Lecture ID filter (String - should auto-cast)
    lecture_id_str = "210"
    print(f"\n4. Filter by lecture_id='{lecture_id_str}' (String - Auto-cast test):")
    docs = await retriever.retrieve(query_en=query, filters={"lecture_id": lecture_id_str})
    print(f"   Found {len(docs)} documents.")

    # 5. Class Name filter
    class_name = "Broadcast Training"
    print(f"\n5. Filter by class_name='{class_name}':")
    docs = await retriever.retrieve(query_en=query, filters={"class_name": class_name})
    print(f"   Found {len(docs)} documents.")

if __name__ == "__main__":
    asyncio.run(test_filters())
