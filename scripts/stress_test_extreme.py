import asyncio
import httpx
import logging
import random
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extreme_stress_test.log")
    ]
)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/chat"

# Scenarios for Extreme Testing
EXTREME_SCENARIOS = [
    {
        "name": "RAPID_FIRE_ATTACK",
        "description": "Send 3 requests for the same session in 500ms to test 409 Conflict.",
        "type": "lock_test"
    },
    {
        "name": "LARGE_PAYLOAD",
        "query": "Explain quantum physics. " * 500, # Very long query
        "lang": "en",
        "type": "load_test"
    },
    {
        "name": "MULTI_LANG_SWITCH",
        "queries": [
            ("Explain cells in English", "en"),
            ("इसे हिंदी में समझाएं", "hi"),
            ("मराठीत सांगा", "mr")
        ],
        "type": "context_test"
    },
    {
        "name": "EDGE_CASE_EMPTY",
        "query": "   ",
        "lang": "en",
        "type": "edge_test"
    },
    {
        "name": "COMPLEX_SOCRATIC",
        "query": "I think the earth is flat. Change my mind using science.",
        "lang": "en",
        "agent_mode": "interactive",
        "type": "educational"
    }
]

async def send_single_query(client: httpx.AsyncClient, payload: Dict[str, Any], label: str):
    start = time.perf_counter()
    try:
        req_start = datetime.now()
        response = await client.post(API_URL, json=payload, timeout=90.0)
        duration = time.perf_counter() - start
        
        status = response.status_code
        result = response.text[:100]
        
        if status == 200:
            logger.info(f"✅ [PASS] {label} | Time: {duration:.2f}s | LLM Calls: {response.json().get('llm_calls')}")
        elif status == 409:
            logger.warning(f"🔒 [LOCK] {label} | Correctly rejected duplicate session.")
        else:
            logger.error(f"❌ [FAIL] {label} | Status: {status} | Error: {result}")
            
        return {"status": status, "duration": duration, "label": label}
    except Exception as e:
        logger.error(f"💥 [ERR] {label} | {str(e)}")
        return {"status": "error", "error": str(e), "label": label}

async def run_lock_test(client: httpx.AsyncClient):
    session_id = f"lock_test_{uuid.uuid4().hex[:6]}"
    payload = {
        "query": "Testing lock...",
        "user_id": "tester_lock",
        "user_session_id": session_id,
        "user_type": "student",
        "language": "en"
    }
    
    logger.info(f"🔥 Starting RAPID_FIRE_ATTACK for session {session_id}")
    tasks = [
        send_single_query(client, payload, f"Rapid_1"),
        send_single_query(client, payload, f"Rapid_2"),
        send_single_query(client, payload, f"Rapid_3")
    ]
    return await asyncio.gather(*tasks)

async def run_context_switch(client: httpx.AsyncClient):
    session_id = f"context_test_{uuid.uuid4().hex[:6]}"
    results = []
    
    logger.info(f"🔄 Starting MULTI_LANG_SWITCH for session {session_id}")
    for i, (q, lang) in enumerate(EXTREME_SCENARIOS[2]["queries"]):
        payload = {
            "query": q,
            "user_id": "tester_multi",
            "user_session_id": session_id,
            "user_type": "student",
            "language": lang
        }
        res = await send_single_query(client, payload, f"Switch_{i}_{lang}")
        results.append(res)
    return results

async def run_extreme_load(client: httpx.AsyncClient, concurrency: int):
    logger.info(f"📈 Starting EXTREME_LOAD with {concurrency} concurrent unique users")
    tasks = []
    for i in range(concurrency):
        payload = {
            "query": f"Educational query {i}: Describe the importance of biodiversity.",
            "user_id": f"user_load_{i}",
            "user_session_id": f"session_load_{i}_{uuid.uuid4().hex[:4]}",
            "user_type": "student",
            "language": random.choice(["en", "hi", "mr"]),
            "student_grade": random.choice(["A", "B", "C"])
        }
        tasks.append(send_single_query(client, payload, f"Load_{i}"))
    return await asyncio.gather(*tasks)

async def main():
    logger.info("🎬 INITIALIZING EXTREME STRESS TEST 🎬")
    async with httpx.AsyncClient() as client:
        # 1. Lock Test (Concurrency Control)
        lock_results = await run_lock_test(client)
        
        # 2. Context Switch Test
        context_results = await run_context_switch(client)
        
        # 3. High Load Test
        load_results = await run_extreme_load(client, 10)
        
        # 4. Large Payload Test
        logger.info("🐘 Sending LARGE_PAYLOAD (Summarization Test)")
        large_payload = {
            "query": EXTREME_SCENARIOS[1]["query"],
            "user_id": "tester_large",
            "user_session_id": f"large_{uuid.uuid4().hex[:4]}",
            "user_type": "student",
            "language": "en"
        }
        large_res = await send_single_query(client, large_payload, "Large_Payload")

    print("\n" + "!"*60)
    print("      EXTREME STRESS TEST FINAL REPORT")
    print("!"*60)
    # Summarize Lock Results
    locks_ok = any(r["status"] == 409 for r in lock_results)
    print(f"Session Locking: {'PASS' if locks_ok else 'FAIL'} (Expected at least one 409)")
    
    # Summarize Success
    total = len(lock_results) + len(context_results) + len(load_results) + 1
    successes = sum(1 for r in lock_results + context_results + load_results + [large_res] if r.get("status") in [200, 409])
    
    print(f"Total Requests:  {total}")
    print(f"Resilience Score: {successes}/{total} ({ (successes/total)*100 :.1f}%)")
    print("!"*60)

if __name__ == "__main__":
    asyncio.run(main())
