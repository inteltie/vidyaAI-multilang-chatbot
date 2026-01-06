import requests
import json
import time

BASE_URL = "http://localhost:8000"
SESSION_ID = f"test_lang_switch_{int(time.time())}"

def chat(query, lang, user_type="student"):
    print(f"\n--- Sending Query: '{query}' [Lang: {lang}] ---")
    payload = {
        "user_session_id": SESSION_ID,
        "user_id": "test_user",
        "user_type": user_type,
        "query": query,
        "language": lang
    }
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Response Language: {data.get('language')}")
        print(f"Message: {data.get('message')[:200]}...") # Print first 200 chars
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

# Turn 1: English
chat("What is the powerhouse of the cell?", "en")

# Turn 2: Marathi (Explicit switch via API)
chat("पेशीचा उर्जा स्त्रोत काय आहे?", "mr")
