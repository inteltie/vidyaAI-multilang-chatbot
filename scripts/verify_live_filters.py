import requests
import json
import uuid

def test_live_filters():
    url = "http://localhost:8001/chat"
    session_id = str(uuid.uuid4())
    
    # Test Case: Provide a class_id and check if only documents with that id are returned.
    # Note: I need to use IDs that likely exist or at least see the logs for the filter.
    payload = {
        "user_session_id": session_id,
        "user_id": "test_user_1",
        "user_type": "student",
        "query": "Explain what is Pythagorean theorem",
        "language": "en",
        "filters": {
            "class_id": 12,
            "additionalProp1": {} # This should be stripped
        },
        "agent_mode": "standard"
    }
    
    print(f"--- Sending request to {url} ---")
    print(f"Filters: {payload['filters']}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"\nResponse Intent: {data.get('intent')}")
        citations = data.get("citations", [])
        print(f"Number of Citations: {len(citations)}")
        
        leaked = []
        for cit in citations:
            meta_class_id = cit.get("class_id")
            print(f"  - Doc ID: {cit.get('id')[:8]}... | class_id: {meta_class_id}")
            if meta_class_id is not None and meta_class_id != 12:
                leaked.append(cit)
        
        if leaked:
            print(f"\n❌ FAILED: Found {len(leaked)} documents with non-matching class_id!")
        elif not citations:
            print("\n⚠️  No citations returned. Check if data for class_id=12 exists.")
        else:
            print("\n✅ PASSED: All citations match class_id=12 (if present).")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_live_filters()
