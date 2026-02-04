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
        logging.FileHandler("multi_user_marathon.log")
    ]
)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/chat"

# --- PERSOAN 1: RAHUL (STUDENT - ENGLISH) ---
RAHUL_TURNS = [
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

# --- PERSONA 2: PRIYA (TEACHER - HINDI) ---
PRIYA_TURNS = [
    "नमस्ते, मैं प्रिया हूं, और मैं भूगोल पढ़ाती हूं।",
    "आज हमें पृथ्वी की परतों (Layers of the Earth) के बारे में चर्चा करनी चाहिए।",
    "मुख्य रूप से पृथ्वी की कितनी परतें होती हैं?",
    "भूपर्पटी (Crust) के बारे में कुछ विस्तार से बताएं।",
    "क्या आप मुझे 'मैंटल' (Mantle) के बारे में भी समझा सकते हैं?",
    "पृथ्वी का केंद्र (Core) किस चीज से बना है?",
    "क्या आप मेरा नाम और मेरा विषय याद रख सकते हैं?",
    "अब अर्थशास्त्र (Economics) पर चलते हैं।",
    "मांग (Demand) का नियम क्या है?",
    "आपूर्ति (Supply) इसे कैसे प्रभावित करती है?",
    "बाजार संतुलन (Market Equilibrium) क्या है?",
    "जीडीपी (GDP) का मतलब क्या होता है?",
    "भूगोल में हमने किस विषय पर बात की थी?",
    "अब इतिहास (History) की बात करते हैं। मौर्य साम्राज्य के संस्थापक कौन थे?",
    "अशोक महान क्यों प्रसिद्ध हैं?",
    "कलिंग युद्ध का उनके जीवन पर क्या प्रभाव पड़ा?",
    "अभी तक हमने जिन तीन विषयों पर चर्चा की है, उनका सारांश दें।",
    "क्या भविष्य के लिए एक अच्छा अध्ययन योजना बना सकते हैं?",
    "एक शिक्षक के रूप में मुझे छात्रों को बेहतर तरीके से कैसे पढ़ाना चाहिए?",
    "अंतिम प्रश्न: मैं कौन हूं और हमने किन विषयों पर चर्चा की?"
]

async def send_turn(client: httpx.AsyncClient, user_id: str, session_id: str, turn: str, turn_num: int, lang: str):
    payload = {
        "query": turn,
        "user_id": user_id,
        "user_session_id": session_id,
        "user_type": "student" if "rahul" in user_id else "teacher",
        "language": lang,
        "agent_mode": "standard"
    }
    
    start = time.perf_counter()
    try:
        response = await client.post(API_URL, json=payload, timeout=90.0)
        duration = time.perf_counter() - start
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ {user_id.upper()} [T{turn_num}] ({duration:.2f}s): {data['message'][:50]}...")
            return data
        else:
            logger.error(f"❌ {user_id.upper()} [T{turn_num}] FAILED: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"💥 {user_id.upper()} [T{turn_num}] ERR: {str(e)}")
        return None

async def run_marathon():
    session_rahul = f"marathon_rahul_{uuid.uuid4().hex[:4]}"
    session_priya = f"marathon_priya_{uuid.uuid4().hex[:4]}"
    
    logger.info(f"🚀 STARTING 20-TURN MULTI-USER MARATHON")
    logger.info(f"👤 Rahul: {session_rahul}")
    logger.info(f"👤 Priya: {session_priya}")
    
    async with httpx.AsyncClient() as client:
        for i in range(20):
            logger.info(f"--- TURN {i+1} / 20 ---")
            # Run Rahul and Priya turns concurrently
            tasks = [
                send_turn(client, "rahul", session_rahul, RAHUL_TURNS[i], i+1, "en"),
                send_turn(client, "priya", session_priya, PRIYA_TURNS[i], i+1, "hi")
            ]
            await asyncio.gather(*tasks)
            # Small stagger to keep it realistic
            await asyncio.sleep(0.5)

async def main():
    await run_marathon()

if __name__ == "__main__":
    asyncio.run(main())
