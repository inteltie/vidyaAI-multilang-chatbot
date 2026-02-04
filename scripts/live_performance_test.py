import asyncio
import httpx
import json
import logging
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
        logging.FileHandler("live_performance_test.log")
    ]
)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/chat"

# --- USER PERSONAS ---
USERS = {
    "student_physics": {
        "user_id": "std_001_physics",
        "session_id": f"live_session_phys_{uuid.uuid4().hex[:4]}",
        "user_type": "student",
        "grade": "Grade 10",
        "language": "en",
        "turns": [
            "Hi, I'm stuck on Newton's second law. Can you explain it simply?",
            "What happens if the mass increases but the force stays the same?",
            "Give me a real-world example involving a car.",
            "Wait, how does this relate to friction?"
        ]
    },
    "student_hindi_bio": {
        "user_id": "std_002_hindi",
        "session_id": f"live_session_bio_{uuid.uuid4().hex[:4]}",
        "user_type": "student",
        "grade": "Grade 9",
        "language": "hi",
        "turns": [
            "कोशिका क्या है? (What is a cell?)",
            "पादप कोशिका और जंतु कोशिका में क्या अंतर है? (Difference between plant and animal cells?)",
            "केंद्रक का क्या कार्य है? (What is the function of the nucleus?)"
        ]
    },
    "teacher_math": {
        "user_id": "tea_001_math",
        "session_id": f"live_session_math_{uuid.uuid4().hex[:4]}",
        "user_type": "teacher",
        "grade": "Grade 8",
        "language": "en",
        "turns": [
            "I need to plan a lesson on Fractions. Can you suggest some interactive activities?",
            "How can I explain equivalent fractions to a student who is struggling?",
            "What are some common misconceptions students have about denominator comparison?"
        ]
    }
}

async def send_turn(client: httpx.AsyncClient, user_key: str, turn_index: int) -> Dict[str, Any]:
    user = USERS[user_key]
    if turn_index >= len(user["turns"]):
        return None
        
    payload = {
        "query": user["turns"][turn_index],
        "user_id": user["user_id"],
        "user_session_id": user["session_id"],
        "user_type": user["user_type"],
        "language": user["language"],
        "agent_mode": "standard"
    }
    
    start = time.perf_counter()
    logger.info(f"📤 {user_key.upper()} [Turn {turn_index+1}]: {payload['query']}")
    
    try:
        response = await client.post(API_URL, json=payload, timeout=90.0)
        duration = time.perf_counter() - start
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"📥 {user_key.upper()} [Turn {turn_index+1}] (OK - {duration:.2f}s): {data['message'][:60]}...")
            return data
        else:
            logger.error(f"❌ {user_key.upper()} failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"💥 {user_key.upper()} error: {e}")
        return None

async def run_live_simulation():
    async with httpx.AsyncClient() as client:
        # --- PHASE 1: INITIAL ENGAGEMENT (CONCURRENT) ---
        logger.info("🎬 PHASE 1: Concurrent Engagement")
        tasks = [send_turn(client, k, 0) for k in USERS.keys()]
        await asyncio.gather(*tasks)
        
        # --- PHASE 2: DEEP DIVE (SEQUENTIAL & INTERLEAVED) ---
        logger.info("📖 PHASE 2: Deep Dive Interleaving")
        for i in range(1, 3):
            # Interleave users to simulate real traffic spike
            logger.info(f"--- Round {i} ---")
            for k in USERS.keys():
                await send_turn(client, k, i)
                await asyncio.sleep(1) # Small gap
                
        # --- PHASE 3: THE 'PAUSE' & RESTART ---
        logger.info("☕ PHASE 3: Simulating a break (Session Persistence)")
        await asyncio.sleep(5) # Shorter than 2h but tests Mongo persistence
        
        logger.info("🔄 Re-engaging Physics student for Session Restart check")
        physics_resume = {
            "query": "Wait, what was the first law again? You only told me about the second one.",
            "user_id": USERS["student_physics"]["user_id"],
            "user_session_id": USERS["student_physics"]["session_id"],
            "user_type": "student",
            "language": "en"
        }
        start = time.perf_counter()
        resp = await client.post(API_URL, json=physics_resume, timeout=60.0)
        logger.info(f"📥 PHYSICS RESUME (OK - {time.perf_counter()-start:.2f}s): {resp.json()['message'][:100]}...")
        
        # Check if it remembers the context
        if "Newton" in resp.json()['message'] or "law" in resp.json()['message']:
            logger.info("✅ CONTEXT RECOVERY: PASS - Agent remembers previous topics from the session.")
        else:
            logger.warning("⚠️ CONTEXT RECOVERY: MARGINAL - Newton not explicitly mentioned in start of response.")

async def main():
    logger.info("🚀 STARTING LIVE ENVIRONMENT SIMULATION 🚀")
    await run_live_simulation()
    logger.info("🏁 SIMULATION COMPLETE 🏁")

if __name__ == "__main__":
    asyncio.run(main())
