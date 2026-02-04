import asyncio
import httpx
import time
import sys

# Test the new Redis-based distributed locking
# Usage: python scripts/verify_session_lock.py [API_URL]

async def test_duplicate_session(url="http://127.0.0.1:8001/chat"):
    session_id = f"test_lock_{int(time.time())}"
    payload = {
        "query": "Hello, please provide a long response to test locking.",
        "user_id": "user_test",
        "user_session_id": session_id,
        "user_type": "student",
        "language": "en"
    }
    
    print(f"🚀 Testing session lock for ID: {session_id}")
    async with httpx.AsyncClient() as client:
        # Start the first request
        print("➡️  Sending Request 1 (should succeed or process)...")
        task1 = asyncio.create_task(client.post(url, json=payload, timeout=90.0))
        
        # Wait a tiny bit to ensure the first one hits Redis
        await asyncio.sleep(0.5)
        
        # Start the second request with the SAME session_id
        print("➡️  Sending Request 2 (should return 409 Conflict)...")
        task2 = asyncio.create_task(client.post(url, json=payload, timeout=90.0))
        
        responses = await asyncio.gather(task1, task2, return_exceptions=True)
        
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"❌ Request {i+1} Exception: {resp}")
            else:
                print(f"📊 Request {i+1} Status: {resp.status_code}")
                if resp.status_code == 409:
                    print(f"✅ SUCCESSFULLY LOCKED: Received expected 409 Conflict for Request {i+1}")
                elif resp.status_code == 200:
                    print(f"✅ SUCCESS: Request {i+1} processed successfully.")

if __name__ == "__main__":
    api_url = "http://127.0.0.1:8001/chat"
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    asyncio.run(test_duplicate_session(api_url))
