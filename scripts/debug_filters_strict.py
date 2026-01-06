import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retriever import RetrieverService
from models import QueryIntent
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_filters():
    load_dotenv()
    
    # Mock settings
    mock_settings = MagicMock()
    mock_settings.embedding_model = "text-embedding-3-large"
    mock_settings.openai_api_key = os.getenv("OPENAI_API_KEY")
    mock_settings.pinecone_api_key = os.getenv("PINECONE_API_KEY")
    mock_settings.pinecone_index = os.getenv("PINECONE_INDEX")
    mock_settings.retriever_top_k = 5
    
    # Instantiate retriever
    retriever = RetrieverService(mock_settings)
    
    # Mock Pinecone Index query method to see what's being sent
    original_query = retriever._index.query
    retriever._index.query = MagicMock(return_value=MagicMock(matches=[]))
    
    print("\n--- Testing Strict Filter Handling ---")
    
    test_cases = [
        {
            "name": "Single integer filter (class_id as int)",
            "filters": {"class_id": 12},
            "expected_pinecone_filter": {"class_id": {"$eq": 12}}
        },
        {
            "name": "Single string filter (class_id as string)",
            "filters": {"class_id": "12"},
            "expected_pinecone_filter": {"class_id": {"$eq": 12}} # Should be casted to int
        },
        {
            "name": "List filter (subject_id as list of strings)",
            "filters": {"subject_id": ["1", "2"]},
            "expected_pinecone_filter": {"subject_id": {"$in": [1, 2]}} # Should be casted to int list
        },
        {
            "name": "Whitelisting check (ignore invalid filters)",
            "filters": {"class_id": 12, "invalid_field": "foo"},
            "expected_pinecone_filter": {"class_id": {"$eq": 12}} # invalid_field should be gone
        }
    ]
    
    for case in test_cases:
        print(f"\nRunning: {case['name']}")
        await retriever.retrieve(
            query_en="test query",
            filters=case["filters"],
            intent=QueryIntent.CONCEPT_EXPLANATION
        )
        
        # Check call arguments
        args, kwargs = retriever._index.query.call_args
        actual_filter = kwargs.get("filter")
        
        print(f"Sent to Pinecone: {actual_filter}")
        if actual_filter == case["expected_pinecone_filter"]:
            print(f"✅ PASSED")
        else:
            print(f"❌ FAILED. Expected: {case['expected_pinecone_filter']}")
            
        retriever._index.query.reset_mock()

if __name__ == "__main__":
    asyncio.run(debug_filters())
