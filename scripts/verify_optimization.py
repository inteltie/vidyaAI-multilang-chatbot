import asyncio
import logging
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.query_classifier import QueryClassifier
from services.context_parser import ContextParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_optimizations():
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    print("\n--- Verifying Query Classifier ---")
    classifier = QueryClassifier(llm)
    
    # 1. Test Heuristics
    test_queries = [
        ("hi", "conversational"),
        ("thanks", "conversational"),
        ("I need help", "conversational"),
        ("i need some help with a topic", "conversational"),
        ("explain photosynthesis", "curriculum_specific"), # Should fail heuristic, go to LLM -> curriculum_specific
    ]
    
    for query, expected_type in test_queries:
        # Test heuristic directly
        heuristic = classifier._check_heuristics(query)
        if expected_type == "conversational":
            if heuristic == "conversational":
                print(f"✅ Heuristic: '{query}' -> {heuristic}")
            else:
                print(f"❌ Heuristic Failed: '{query}' -> {heuristic} (Expected: {expected_type})")
        else:
            if heuristic is None:
                print(f"✅ Heuristic Bypassed: '{query}'")
            else:
                print(f"❌ Heuristic False Positive: '{query}' -> {heuristic}")

    # 2. Test LLM Classification (Structured Output)
    print("\n--- Testing LLM Classification ---")
    llm_query = "explain photosynthesis"
    result = await classifier.classify(llm_query, [])
    print(f"Query: '{llm_query}' -> {result}")
    if result == "curriculum_specific":
        print("✅ LLM Classification correct.")
    else:
        print(f"❌ LLM Classification failed. Got: {result}")

    print("\n--- Verifying Context Parser ---")
    parser = ContextParser(llm)
    
    # Test extraction
    history = [{"role": "user", "content": "I am in class 10 studying math from lecture 101"}]
    meta = await parser.extract_from_history(history)
    print(f"Extracted from history: {meta}")
    
    if meta.get("class_level") in ["10", "Class 10"] and meta.get("subject") in ["math", "Math"]:
        print("✅ Context Parser extraction correct.")
    else:
        print("❌ Context Parser extraction failed.")

if __name__ == "__main__":
    asyncio.run(verify_optimizations())
