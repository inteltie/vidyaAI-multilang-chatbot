
import requests
import json
import time

def test_llm_call_accuracy():
    base_url = "http://localhost:8001"
    
    # Use a query that triggers curriculum retrieval and agent reasoning
    payload = {
        "user_id": "test_llm_user",
        "user_session_id": "test_llm_session",
        "user_type": "student",
        "query": "What is friction? Explain its types.",
        "student_grade": "A",
        "language": "hi" # Force translation to add +1 call
    }
    
    print("--- Testing LLM Call Count Accuracy ---")
    print(f"Query: {payload['query']} (Language: {payload['language']})")
    
    start_time = time.time()
    response = requests.post(f"{base_url}/chat", json=payload)
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Total Request Time: {end_time - start_time:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        llm_calls = data.get('llm_calls', 0)
        print(f"\nReported LLM Calls: {llm_calls}")
        
        print("\n--- Expected Breakdown (Minimum) ---")
        print("1. Analyze Query (Classifier): 1")
        print("2. Agent Reasoning (ReAct): 1+ (at least one for final answer)")
        print("3. Groundedness Check (Guardian): 1")
        print("4. Response Translation (Translator): 1")
        print("TOTAL EXPECTED: >= 4")
        
        if llm_calls >= 4:
            print(f"\nSUCCESS: LLM call count ({llm_calls}) seems accurate and cumulative.")
        else:
            print(f"\nWARNING: LLM call count ({llm_calls}) might be under-counting. Expected at least 4.")
            
        # Check message content to see if it's translated
        print(f"\nResponse sample (first 100 chars): {data['message'][:100]}...")
        
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_llm_call_accuracy()
