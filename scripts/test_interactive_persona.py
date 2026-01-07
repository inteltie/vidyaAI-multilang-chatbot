import requests
import json
import uuid

def test_interactive_persona():
    url = "http://localhost:8001/chat"
    grades = ["A", "B", "C", "D"]
    query = "Why do balls roll down hills?"
    
    print(f"--- Testing INTERACTIVE Persona System ---")
    
    for grade in grades:
        session_id = str(uuid.uuid4())
        payload = {
            "user_session_id": session_id,
            "user_id": f"test_user_{grade}_interactive",
            "user_type": "student",
            "query": query,
            "language": "en",
            "student_grade": grade,
            "agent_mode": "interactive"
        }
        
        print(f"\n[GRADE: {grade.upper()}] Sending request...")
        try:
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", "")
            print(f"--- RESPONSE START ---\n{message[:1500]}\n--- RESPONSE END ---")
        except Exception as e:
            print(f"Error for {grade}: {e}")

if __name__ == "__main__":
    test_interactive_persona()
