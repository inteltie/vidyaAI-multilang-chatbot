#!/usr/bin/env python3
"""
Test script for multi-turn conversation, timing, and Redis memory persistence.
Uses only standard library (urllib) and redis-py for maximum compatibility.
"""

import json
import time
import uuid
import urllib.request
import urllib.error
import redis
from typing import List, Dict, Any

# Configuration
BASE_URL = "http://localhost:8001"
REDIS_URL = "redis://localhost:6379"
USER_ID = "test_student_001"
SESSION_ID = f"test_session_{uuid.uuid4().hex[:8]}"
STUDENT_GRADE = "B"

# Test cases: A string of related questions
TEST_TURNS = [
    {
        "query": "Hi, I'm Rahul from Class 10. I want to learn about Science.",
        "expected_mention": "Rahul",
        "description": "Initial greeting and context setting"
    },
    {
        "query": "Can you explain what is photosynthesis?",
        "expected_mention": "photosynthesis",
        "description": "Subject-specific query"
    },
    {
        "query": "What are the main requirements for this process?",
        "expected_mention": "light", # or chlorophyll, CO2, water
        "description": "Follow-up query (requires memory of 'photosynthesis')"
    },
    {
        "query": "Wait, what was my name again?",
        "expected_mention": "Rahul",
        "description": "Memory check for personal context"
    }
]

def check_redis_memory(session_id: str):
    """Check the contents of Redis for the given session."""
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        # Use the buffer key pattern
        key = f"chat:{session_id}:buffer"
        messages = client.lrange(key, 0, -1)
        
        if not messages:
            # Try the other common pattern
            key = f"chat:{session_id}"
            messages = client.lrange(key, 0, -1)
            
        print(f"\n[REDIS] Key: {key} | Messages found: {len(messages)}")
        for i, msg in enumerate(messages):
            try:
                data = json.loads(msg)
                role = data.get('role', '???').upper()
                content = data.get('content', '')[:50]
                print(f"  {i+1}. {role}: {content}...")
            except:
                print(f"  {i+1}. [Raw]: {msg[:50]}...")
        client.close()
    except Exception as e:
        print(f"[REDIS ERROR] {e}")

def run_test():
    print(f"🚀 Starting Multi-turn Test Session: {SESSION_ID}")
    print(f"👤 User: {USER_ID} (Grade {STUDENT_GRADE})")
    print("-" * 60)

    for i, turn in enumerate(TEST_TURNS):
        print(f"\n▶ TURN {i+1}: {turn['description']}")
        print(f"USER: {turn['query']}")
        
        payload = {
            "user_session_id": SESSION_ID,
            "user_id": USER_ID,
            "user_type": "student",
            "query": turn["query"],
            "student_grade": STUDENT_GRADE,
            "language": "en"
        }
        
        data_bytes = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/chat",
            data=data_bytes,
            headers={'Content-Type': 'application/json'}
        )
        
        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req) as response:
                duration = time.perf_counter() - start_time
                resp_data = json.loads(response.read().decode('utf-8'))
                
                print(f"🤖 VIDYA ({duration:.2f}s): {resp_data['message'][:150]}...")
                print(f"📍 Intent: {resp_data.get('intent')} | LLM Calls: {resp_data.get('llm_calls')}")
                
                citations = resp_data.get("citations", [])
                if citations:
                    print(f"📚 Citations found: {len(citations)}")
                else:
                    print("📚 No citations")
                
                # Verify expected mentions
                lowered_resp = resp_data['message'].lower()
                if turn['expected_mention'].lower() in lowered_resp:
                    print(f"✅ Found expected info: '{turn['expected_mention']}'")
                else:
                    print(f"❌ Missing expected info: '{turn['expected_mention']}'")
                    
        except urllib.error.HTTPError as e:
            print(f"❌ API Error {e.code}: {e.read().decode()}")
            break
        except Exception as e:
            print(f"❌ Request failed: {e}")
            break
            
        # Check Redis
        check_redis_memory(SESSION_ID)
        print("-" * 40)

    print("\n🏁 Test completed.")

if __name__ == "__main__":
    run_test()
