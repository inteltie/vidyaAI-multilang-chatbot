import asyncio
import httpx
import json
import logging
import random
import time
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("load_test_results.log")
    ]
)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/chat"

# Scenarios for testing
SCENARIOS = [
    {
        "name": "Social Greeting - Hindi",
        "payload": {
            "query": "नमस्ते विद्या, आप कैसी हैं?",
            "user_id": "user_hi_001",
            "user_session_id": f"session_hi_{int(time.time())}",
            "user_type": "student",
            "language": "hi",
            "agent_mode": "standard"
        }
    },
    {
        "name": "Social Greeting - Marathi",
        "payload": {
            "query": "नमस्कार विद्या, तू कशी आहेस?",
            "user_id": "user_mr_002",
            "user_session_id": f"session_mr_{int(time.time())}",
            "user_type": "student",
            "language": "mr",
            "agent_mode": "standard"
        }
    },
    {
        "name": "Science Question - English",
        "payload": {
            "query": "What is photosynthesis?",
            "user_id": "user_en_003",
            "user_session_id": f"session_en_{int(time.time())}",
            "user_type": "student",
            "language": "en",
            "agent_mode": "standard"
        }
    },
    {
        "name": "History Question - Hindi",
        "payload": {
            "query": "भारत की आजादी में महात्मा गांधी का क्या योगदान था?",
            "user_id": "user_hi_004",
            "user_session_id": f"session_hi_edu_{int(time.time())}",
            "user_type": "student",
            "language": "hi",
            "agent_mode": "standard"
        }
    },
    {
        "name": "Socratic Interaction - English",
        "payload": {
            "query": "I don't understand how gravity works.",
            "user_id": "user_en_005",
            "user_session_id": f"session_en_soc_{int(time.time())}",
            "user_type": "student",
            "language": "en",
            "agent_mode": "interactive"
        }
    },
    {
        "name": "Teacher Support - English",
        "payload": {
            "query": "How can I explain the concept of energy to 10th grade students?",
            "user_id": "teacher_en_006",
            "user_session_id": f"session_tea_en_{int(time.time())}",
            "user_type": "teacher",
            "language": "en",
            "agent_mode": "standard"
        }
    }
]

async def send_request(client: httpx.AsyncClient, scenario: Dict[str, Any], index: int) -> Dict[str, Any]:
    name = scenario["name"]
    payload = scenario["payload"].copy()
    # Ensure unique session ID for concurrency
    payload["user_session_id"] = f"{payload['user_session_id']}_{index}_{random.randint(100, 999)}"
    
    start_time = time.perf_counter()
    try:
        logger.info(f"🚀 Sending request {index}: {name} (Lang: {payload['language']})")
        response = await client.post(API_URL, json=payload, timeout=90.0)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Success {index}: {name} | Duration: {duration:.2f}s | LLM Calls: {data.get('llm_calls')}")
            # Check language consistency
            resp_msg = data.get("message", "")
            resp_lang = data.get("language", "")
            logger.info(f"📝 Response {index} ({resp_lang}): {resp_msg[:50]}...")
            return {
                "name": name,
                "status": "success",
                "duration": duration,
                "llm_calls": data.get("llm_calls"),
                "language": resp_lang
            }
        else:
            logger.error(f"❌ Failed {index}: {name} | Status: {response.status_code} | Error: {response.text}")
            return {"name": name, "status": "failed", "error": response.text}
    except Exception as e:
        logger.error(f"💥 Exception {index}: {name} | Error: {str(e)}")
        return {"name": name, "status": "error", "error": str(e)}

async def run_load_test(concurrency: int = 5):
    logger.info(f"🏁 Starting Production Load Test with concurrency={concurrency}")
    
    async with httpx.AsyncClient() as client:
        # Pick diverse scenarios
        test_batch = []
        for i in range(concurrency):
            test_batch.append(SCENARIOS[i % len(SCENARIOS)])
        
        # Run concurrently
        tasks = [send_request(client, scenario, i) for i, scenario in enumerate(test_batch)]
        results = await asyncio.gather(*tasks)
        
    # Print Summary
    print("\n" + "="*50)
    print("      PRODUCTION LOAD TEST SUMMARY")
    print("="*50)
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"Total Requests: {len(results)}")
    print(f"Successful:     {success_count}")
    print(f"Failed:         {len(results) - success_count}")
    
    if success_count > 0:
        avg_duration = sum(r["duration"] for r in results if r["status"] == "success") / success_count
        print(f"Avg Duration:   {avg_duration:.2f}s")
    print("="*50)

if __name__ == "__main__":
    import sys
    conc = 3
    if len(sys.argv) > 1:
        conc = int(sys.argv[1])
    asyncio.run(run_load_test(conc))
