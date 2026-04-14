import requests
import uuid
import json
import sys

def chat():
    url = "http://localhost:8001/chat"
    user_id = "human_tester_01"
    session_id = f"session_{uuid.uuid4().hex[:6]}"
    
    print(f"--- VidyaAI Interactive Chat ---")
    print(f"Session: {session_id}")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            payload = {
                "user_session_id": session_id,
                "user_id": user_id,
                "user_type": "student",
                "query": query,
                "language": "en",
                "student_grade": "B"
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"\nVidya: {data['message']}")
                print(f"--- Metadata ---")
                print(f"Intent: {data['intent']} | LLM Calls: {data['llm_calls']}")
                print(f"Tokens: Primary(In:{data['primary_input_tokens']}, Out:{data['primary_output_tokens']}) | Aux(In:{data['aux_input_tokens']}, Out:{data['aux_output_tokens']})")
                if data.get('background_total_tokens', 0) > 0:
                    print(f"Background (Nano): {data['background_total_tokens']} tokens")
                print(f"----------------\n")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    chat()
