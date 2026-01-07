import requests
import json
import uuid

def test_persona_system():
    url = "http://localhost:8001/chat"
    
    grades = ["A", "B", "C", "D"]
    modes = ["standard", "interactive"]
    query = "Explain what is friction"
    
    print(f"--- Testing Grade-Based Persona System ---")
    print(f"Query: {query}\n")
    
    for mode in modes:
        print(f"\n{'='*20} MODE: {mode.upper()} {'='*20}")
        for grade in grades:
            session_id = str(uuid.uuid4())
            payload = {
                "user_session_id": session_id,
                "user_id": f"test_user_{grade}_{mode}",
                "user_type": "student",
                "query": query,
                "language": "en",
                "student_grade": grade,
                "agent_mode": mode
            }
            
            print(f"\n[GRADE: {grade.upper()}] Sending request...")
            try:
                response = requests.post(url, json=payload, timeout=90)
                response.raise_for_status()
                data = response.json()
                
                message = data.get("message", "")
                print(f"--- RESPONSE START ---")
                print(message[:1000]) # Increased character limit to see full impact
                print(f"--- RESPONSE END ---")
                
            except Exception as e:
                print(f"Error for {grade} ({mode}): {e}")

if __name__ == "__main__":
    test_persona_system()

if __name__ == "__main__":
    test_persona_system()
