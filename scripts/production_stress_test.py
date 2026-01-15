import asyncio
import httpx
import json
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/chat"

TIME_STR = datetime.now().strftime('%H%M%S')
USERS = {
    "aisha": {
        "user_id": "student_aisha_001",
        "session_id": f"session_aisha_{TIME_STR}",
        "user_type": "student",
        "grade": "Grade 9",
        "name": "Aisha"
    },
    "vikram": {
        "user_id": "teacher_vikram_002",
        "session_id": f"session_vikram_{TIME_STR}",
        "user_type": "teacher",
        "grade": "Grade 10",
        "name": "Vikram"
    }
}

async def send_query(client, user_data, query):
    payload = {
        "query": query,
        "user_id": user_data["user_id"],
        "user_session_id": user_data["session_id"],
        "user_type": user_data["user_type"],
        "grade": user_data["grade"]
    }
    
    try:
        resp = await client.post(API_URL, json=payload, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Post failed {resp.status_code}: {resp.text}")
        return None
    except Exception as e:
        logger.error(f"Exception in send_query: {e}")
        return None

async def run_simulation():
    async with httpx.AsyncClient() as client:
        print("\n🚀 Production Simulation: Multi-User session Persistence")
        print("=" * 60)

        # 1. Aisha Turn 1
        print("\n[AISHA] -> Hi, I'm Aisha, a Grade 9 student.")
        r_a1 = await send_query(client, USERS["aisha"], "Hi, I'm Aisha, a Grade 9 student.")
        print(f"VIDYA: {r_a1.get('response') if r_a1 else 'FAILED'}")

        # 2. Vikram Turn 1
        print("\n[VIKRAM] -> Hello, I'm Mr. Vikram, a Physics teacher.")
        r_v1 = await send_query(client, USERS["vikram"], "Hello, I'm Mr. Vikram, a Physics teacher.")
        print(f"VIDYA: {r_v1.get('response') if r_v1 else 'FAILED'}")

        # 3. Aisha Turn 2
        print("\n[AISHA] -> Tell me about cell biology.")
        r_a2 = await send_query(client, USERS["aisha"], "Tell me about cell biology.")
        print(f"VIDYA: {r_a2.get('response') if r_a2 else 'FAILED'}")

        # 4. Vikram Turn 2
        print("\n[VIKRAM] -> I need help explaining Newton's laws of motion.")
        r_v2 = await send_query(client, USERS["vikram"], "I need help explaining Newton's laws of motion.")
        print(f"VIDYA: {r_v2.get('response') if r_v2 else 'FAILED'}")

        # 5. Isolation/Memory Check
        print("\n" + "-"*20 + " CROSS-USER MEMORY CHECK " + "-"*20)
        
        print("\n[AISHA] -> Who am I and what was my first question?")
        r_a3 = await send_query(client, USERS["aisha"], "Who am I and what was my first question?")
        if r_a3:
            resp_a = r_a3.get('response', '')
            print(f"VIDYA (to Aisha): {resp_a}")
            a_name_ok = "Aisha" in resp_a
            a_context_ok = any(x in resp_a.lower() for x in ["hi", "greeting", "aisha", "grade 9"])
            print(f">> AISHA VERDICT: Name={'PASS' if a_name_ok else 'FAIL'}, Context={'PASS' if a_context_ok else 'FAIL'}")
        else:
            print("AISHA TURN 3 FAILED")

        print("\n[VIKRAM] -> Who am I and what is my subject?")
        r_v3 = await send_query(client, USERS["vikram"], "Who am I and what is my subject?")
        if r_v3:
            resp_v = r_v3.get('response', '')
            print(f"VIDYA (to Vikram): {resp_v}")
            v_name_ok = "Vikram" in resp_v
            v_context_ok = any(x in resp_v.lower() for x in ["teacher", "physics", "newton", "motion"])
            print(f">> VIKRAM VERDICT: Name={'PASS' if v_name_ok else 'FAIL'}, Context={'PASS' if v_context_ok else 'FAIL'}")
        else:
            print("VIKRAM TURN 3 FAILED")

        print("\n" + "-"*60)
        print("🏁 Simulation Finished.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
