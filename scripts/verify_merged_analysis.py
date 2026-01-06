import asyncio
import logging
import os
import sys
from time import perf_counter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.query_classifier import QueryClassifier, QueryClassification
from nodes.analyze_query import AnalyzeQueryNode
from state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_merged_analysis():
    load_dotenv()
    
    # Use gpt-4o-mini for testing as per optimization
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classifier = QueryClassifier(llm)
    
    # Mock Retriever (similar to the one in nodes/analyze_query.py)
    class MockRetriever:
        async def retrieve(self, **kwargs):
            return [{"id": "doc1", "score": 0.9, "content": "Photosynthesis content"}]
            
    node = AnalyzeQueryNode(classifier, MockRetriever())
    
    print("\n--- Testing Merged Analysis Node ---")
    
    # Test Case 1: Educational query with metadata
    state: AgentState = {
        "query": "I am in class 10 and want to learn about photosynthesis for my science chapter 'Plant Nutrition' from lecture 42",
        "conversation_history": [],
        "session_metadata": {},
        "llm_calls": 0,
        "timings": {}
    }
    
    print(f"\nQuery: {state['query']}")
    start = perf_counter()
    new_state = await node(state)
    total_node_time = perf_counter() - start
    
    print(f"Node Duration: {new_state['timings']['analyze_query']:.3f}s")
    print(f"Total Call Duration: {total_node_time:.3f}s")
    print(f"LLM Calls: {new_state['llm_calls']}")
    
    print("\nExtraction Results:")
    meta = new_state.get("session_metadata", {})
    print(f"Class Level: {meta.get('class_level')}")
    print(f"Subject: {meta.get('subject')}")
    print(f"Topics (Chapter): {meta.get('topics')}")
    print(f"Lecture ID: {meta.get('lecture_id')}")
    
    valid_extraction = (
        meta.get("class_level") is not None and
        meta.get("topics") is not None and
        meta.get("lecture_id") is not None
    )
    
    if valid_extraction:
        print("✅ Merged Context Extraction successful.")
    else:
        print("❌ Merged Context Extraction missing fields.")

    # Test Case 2: Conversational query
    state_conv: AgentState = {
        "query": "Hello how are you?",
        "conversation_history": [],
        "session_metadata": {},
        "llm_calls": 0,
        "timings": {}
    }
    print(f"\nQuery: {state_conv['query']}")
    new_state_conv = await node(state_conv)
    print(f"Query Type: {new_state_conv['query_type']}")
    if new_state_conv['query_type'] == "conversational":
        print("✅ Conversational classification successful.")
    else:
        print("❌ Conversational classification failed.")

    # Test Case 3: Verify Proactive RAG Prefill
    if new_state.get("prefilled_observations"):
        print("✅ Proactive RAG prefill works.")
        print(f"RAG Quality: {new_state.get('rag_quality')}")
    else:
        print("❌ Proactive RAG prefill failed.")

if __name__ == "__main__":
    asyncio.run(verify_merged_analysis())
