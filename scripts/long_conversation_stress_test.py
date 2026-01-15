import asyncio
import httpx
import json
import logging
import time
from datetime import datetime
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/chat"

SESSION_DATA = {
    "user_id": "rahul_stress_test",
    "user_session_id": f"stress_session_{datetime.now().strftime('%H%M%S')}",
    "user_type": "student",
    "grade": "Grade 10"
}

QUERIES = [
    "Hi, I'm Rahul, a Class 10 student.",
    "I want to learn about Physics today.",
    "What is motion?",
    "Tell me about inertia.",
    "Does mass affect it?",
    "What about Newton's First Law?",
    "Wait, what's my name again?",
    "Actually, let's talk about Chemistry.",
    "What is an atom?",
    "What are subatomic particles?",
    "Who discovered the electron?",
    "What is the Rutherford model?",
    "What physics topic were we discussing earlier?",
    "Moving to Biology. What is DNA?",
    "What are its building blocks?",
    "How does it replicate?",
    "Summarize what we have discussed in Physics and Chemistry briefly.",
    "Tell me a fun science fact about space.",
    "How many tokens am I allowed to use in a response?",
    "Final check: Who am I and which three subjects did we discuss?"
]

async def send_query(client, query, turn):
    payload = {
        "query": query,
        **SESSION_DATA
    }
    
    start_time = time.perf_counter()
    try:
        resp = await client.post(API_URL, json=payload, timeout=60.0)
        duration = time.perf_counter() - start_time
        
        if resp.status_code == 200:
            result = resp.json()
            return {
                "turn": turn,
                "query": query,
                "response": result.get("response", ""),
                "duration": duration,
                "llm_calls": result.get("llm_calls", 0),
                "intent": result.get("intent", "n/a"),
                "status": "SUCCESS"
            }
        else:
            logger.error(f"Turn {turn} FAILED: {resp.status_code} - {resp.text}")
            return {"turn": turn, "status": "ERROR", "error": resp.status_code}
    except Exception as e:
        logger.error(f"Turn {turn} EXCEPTION: {e}")
        return {"turn": turn, "status": "EXCEPTION", "error": str(e)}

async def run_stress_test():
    print(f"\n🚀 Starting 20-Turn Long Conversation Stress Test")
    print(f"👤 User: Rahul | Session: {SESSION_DATA['user_session_id']}")
    print("=" * 70)

    results = []
    async with httpx.AsyncClient() as client:
        for i, query in enumerate(QUERIES, 1):
            print(f"[{i}/20] Sending: {query[:50]}...")
            res = await send_query(client, query, i)
            
            if res["status"] == "SUCCESS":
                print(f"   ✅ Done in {res['duration']:.2f}s | Intent: {res['intent']} | LLM Calls: {res['llm_calls']}")
                results.append(res)
            else:
                print(f"   ❌ FAILED")
                results.append(res)
            
            # Small delay to mimic human reading and avoid overwhelming the API
            await asyncio.sleep(0.5)

    # Performance Analysis
    durations = [r["duration"] for r in results if r["status"] == "SUCCESS"]
    llm_calls = [r["llm_calls"] for r in results if r["status"] == "SUCCESS"]
    
    avg_dur = statistics.mean(durations) if durations else 0
    max_dur = max(durations) if durations else 0
    min_dur = min(durations) if durations else 0
    total_llm = sum(llm_calls)

    print("\n" + "="*70)
    print("📊 PERFORMANCE REPORT")
    print("-" * 70)
    print(f"Total Turns:       {len(results)}")
    print(f"Average Resp Time:  {avg_dur:.2f}s")
    print(f"Peak Resp Time:     {max_dur:.2f}s")
    print(f"Min Resp Time:      {min_dur:.2f}s")
    print(f"Total LLM Calls:    {total_llm}")
    print(f"Avg LLM Per Turn:   {total_llm/len(results):.2f}")
    
    print("\n📝 MEMORY/IDENTITIY VERIFICATION")
    print("-" * 70)
    
    # Check specific turns for memory
    turn_7 = next((r for r in results if r["turn"] == 7), None)
    if turn_7:
        name_ok = "Rahul" in turn_7["response"]
        print(f"Turn 7 (Name Check):  {'PASS' if name_ok else 'FAIL'}")
    
    turn_13 = next((r for r in results if r["turn"] == 13), None)
    if turn_13:
        phys_ok = "physics" in turn_13["response"].lower() or "motion" in turn_13["response"].lower()
        print(f"Turn 13 (Context Shift): {'PASS' if phys_ok else 'FAIL'}")
        
    turn_20 = next((r for r in results if r["turn"] == 20), None)
    if turn_20:
        final_ok = "Rahul" in turn_20["response"] and all(x in turn_20["response"].lower() for x in ["physics", "chemistry", "biology"])
        print(f"Turn 20 (Final Memory): {'PASS' if final_ok else 'FAIL'}")
        if not final_ok:
            print(f"   [DEBUG] Turn 20 Response: {turn_20['response'][:100]}...")

    print("=" * 70)
    print("🏁 Stress Test Finished.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
