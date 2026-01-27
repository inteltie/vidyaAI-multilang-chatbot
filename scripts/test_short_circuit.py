import json
import time
import urllib.request

BASE_URL = "http://localhost:8001"

def test_short_circuit():
    queries = ["ok", "thanks", "hi"]
    for query in queries:
        print(f"\nTesting: '{query}'")
        payload = {
            "user_session_id": "short_circuit_test",
            "user_id": "test_user",
            "user_type": "student",
            "query": query,
            "language": "en"
        }
        data_bytes = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/chat", data=data_bytes, headers={'Content-Type': 'application/json'})
        
        start = time.perf_counter()
        with urllib.request.urlopen(req) as response:
            duration = time.perf_counter() - start
            resp_data = json.loads(response.read().decode('utf-8'))
            print(f"Response: {resp_data['message'][:50]}...")
            print(f"Time: {duration:.3f}s")
            print(f"LLM Calls: {resp_data.get('llm_calls')}")

if __name__ == "__main__":
    test_short_circuit()
