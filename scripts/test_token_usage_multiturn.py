#!/usr/bin/env python3
"""
Multi-turn token usage test (Hindi + English + Marathi).

Runs >20 turns against /chat, then generates a report that shows
approximate token usage per turn (user + assistant text only).

Note: This is an approximation. Server-side token usage includes
system prompts, tool calls, and history, which are not counted here.
"""

import json
import os
import time
from datetime import datetime
import uuid
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Dict, Any

import tiktoken


BASE_URL = os.getenv("BASE_URL", "http://localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
DEFAULT_USER_ID = os.getenv("USER_ID", "test_student_001")
DEFAULT_SESSION_ID = os.getenv("SESSION_ID", f"token_test_{uuid.uuid4().hex[:8]}")
STUDENT_GRADE = os.getenv("STUDENT_GRADE", "B")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "reports")
LOG_FILE = os.getenv("LOG_FILE", "")
DEBUG_RESPONSE = os.getenv("DEBUG_RESPONSE", "").lower() in {"1", "true", "yes"}


def _parse_log_totals(log_path: str, since_epoch: float) -> List[Dict[str, int]]:
    """
    Parse log file for '[TOKEN_USAGE] TOTAL REQUEST CYCLE' lines after since_epoch.
    Returns a list of dicts in chronological order.
    """
    if not log_path:
        return []
    if not os.path.exists(log_path):
        return []

    results = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "[TOKEN_USAGE] TOTAL REQUEST CYCLE:" not in line:
                    continue
                # Parse timestamp at line start: "YYYY-MM-DD HH:MM:SS,mmm"
                try:
                    ts_str = line.split(" [", 1)[0].strip()
                    log_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                    log_ts = log_dt.timestamp()
                except Exception:
                    log_ts = None

                if log_ts is not None and log_ts < since_epoch:
                    continue

                # Extract token counts
                # Example: input_tokens=3154, output_tokens=258, total_tokens=3412
                try:
                    part = line.split("TOTAL REQUEST CYCLE:", 1)[1]
                    fields = part.split(",")
                    it = int(fields[0].split("=", 1)[1].strip())
                    ot = int(fields[1].split("=", 1)[1].strip())
                    tt = int(fields[2].split("=", 1)[1].strip())
                    results.append({"input_tokens": it, "output_tokens": ot, "total_tokens": tt})
                except Exception:
                    continue
    except Exception:
        return []

    return results


@dataclass
class Turn:
    query: str
    description: str


CONVERSATIONS = [
    {
        "user_id": "student_en_chem",
        "session_id": f"token_test_en_{uuid.uuid4().hex[:6]}",
        "language": "en",
        "turns": [
            Turn("Hi, I am Riya from Class 12. I need help with chemistry.", "English greeting and context"),
            Turn("Define chemical kinetics and the rate of reaction.", "Kinetics definition"),
            Turn("Explain the rate law and rate constant with an example.", "Kinetics rate law"),
            Turn("What is the Arrhenius equation? Explain terms.", "Arrhenius equation"),
            Turn("How does temperature affect reaction rate? Use Arrhenius reasoning.", "Temperature effect"),
            Turn("Explain activation energy and how catalysts lower it.", "Activation energy"),
            Turn("Distinguish between zero, first, and second order reactions.", "Order of reaction"),
            Turn("For a first order reaction, derive half-life expression.", "Half-life"),
            Turn("How do you determine reaction order experimentally?", "Experimental determination"),
            Turn("Explain collision theory and transition state theory.", "Kinetics theory"),
            Turn("Now switch to neural networks: define perceptron.", "NN topic shift"),
            Turn("Explain backpropagation in simple steps.", "Backpropagation"),
            Turn("What is the role of activation functions like ReLU and sigmoid?", "Activation functions"),
            Turn("Explain overfitting and how dropout helps.", "Regularization"),
            Turn("Compare CNNs and RNNs with use cases.", "Architecture comparison"),
            Turn("Now psychology: explain classical conditioning with Pavlov.", "Classical conditioning"),
            Turn("Differentiate unconditioned vs conditioned stimulus.", "Stimulus differentiation"),
            Turn("Explain extinction and spontaneous recovery.", "Learning effects"),
            Turn("Humanities: Who was Akbar and what is Sulh-i-Kul?", "History"),
            Turn("Explain the Renaissance and its impact on Europe.", "Renaissance"),
            Turn("Summarize the key topics in 5 bullet points.", "Summary request"),
            Turn("Give 3 exam-style questions from these topics.", "Exam questions"),
        ],
    },
    {
        "user_id": "student_hi_mixed",
        "session_id": f"token_test_hi_{uuid.uuid4().hex[:6]}",
        "language": "hi",
        "turns": [
            Turn("नमस्ते, मैं आरव हूँ और मुझे विज्ञान समझना है।", "Hindi greeting"),
            Turn("रासायनिक गतिकी क्या है? दर नियम समझाइए।", "Kinetics definition"),
            Turn("Arrhenius समीकरण लिखिए और उसके पदों का अर्थ बताइए।", "Arrhenius"),
            Turn("तापमान बढ़ने पर दर कैसे बदलती है?", "Temperature effect"),
            Turn("उत्प्रेरक सक्रियण ऊर्जा को कैसे कम करता है?", "Catalyst"),
            Turn("प्रथम क्रम की अभिक्रिया का अर्द्ध-आयु सूत्र बताइए।", "Half-life"),
            Turn("अब न्यूरल नेटवर्क पर आते हैं: परसेप्ट्रॉन क्या है?", "NN perceptron"),
            Turn("बैकप्रोपेगेशन क्या है? सरल चरणों में समझाइए।", "Backprop"),
            Turn("ReLU और sigmoid के बीच अंतर बताइए।", "Activation functions"),
            Turn("ओवरफिटिंग क्या है और ड्रॉपआउट कैसे मदद करता है?", "Regularization"),
            Turn("अब मनोविज्ञान: शास्त्रीय अनुबंधन (classical conditioning) क्या है?", "Classical conditioning"),
            Turn("अनियंत्रित एवं नियंत्रित उद्दीपन का अंतर बताइए।", "Stimulus"),
            Turn("विलुप्ति और स्वस्फूर्त पुनरुत्थान समझाइए।", "Learning effects"),
            Turn("मानविकी: अकबर और दीन-ए-इलाही के बारे में बताइए।", "History"),
            Turn("पुनर्जागरण क्या था और इसका प्रभाव क्या था?", "Renaissance"),
            Turn("सभी विषयों का 4 बिंदुओं में सार दीजिए।", "Summary"),
            Turn("इन विषयों पर 3 अभ्यास प्रश्न दीजिए।", "Practice questions"),
            Turn("एक संक्षिप्त निष्कर्ष लिखिए।", "Conclusion"),
            Turn("कक्षा 12 के लिए कठिनाई स्तर बताइए।", "Difficulty"),
            Turn("रिवीजन के लिए 3 टिप्स दीजिए।", "Revision tips"),
            Turn("अब एक त्वरित पुनरावृत्ति करें।", "Quick recap"),
        ],
    },
    {
        "user_id": "student_mr_mixed",
        "session_id": f"token_test_mr_{uuid.uuid4().hex[:6]}",
        "language": "mr",
        "turns": [
            Turn("नमस्कार, मी ईशान आहे. मला विज्ञान शिकायचे आहे.", "Marathi greeting"),
            Turn("रासायनिक गतिकी म्हणजे काय? दर नियम समजावून सांगा.", "Kinetics definition"),
            Turn("Arrhenius समीकरण लिहा आणि त्यातील घटक समजावून सांगा.", "Arrhenius"),
            Turn("तापमान वाढल्यावर अभिक्रियेचा दर कसा बदलतो?", "Temperature effect"),
            Turn("उत्प्रेरक सक्रियण ऊर्जा कशी कमी करतो?", "Catalyst"),
            Turn("प्रथम क्रम अभिक्रियेचा अर्धायुष्याचा सूत्र द्या.", "Half-life"),
            Turn("आता न्यूरल नेटवर्क: परसेप्ट्रॉन काय आहे?", "NN perceptron"),
            Turn("बॅकप्रोपेगेशन सोप्या टप्प्यांत समजावून सांगा.", "Backprop"),
            Turn("ReLU आणि sigmoid यात फरक काय?", "Activation functions"),
            Turn("ओव्हरफिटिंग काय आहे आणि ड्रॉपआउट कसे मदत करते?", "Regularization"),
            Turn("आता मानसशास्त्र: classical conditioning समजावून सांगा.", "Classical conditioning"),
            Turn("unconditioned आणि conditioned stimulus मध्ये फरक काय?", "Stimulus"),
            Turn("extinction आणि spontaneous recovery समजावून सांगा.", "Learning effects"),
            Turn("मानविकी: अकबर आणि सुल्ह-ए-कुल बद्दल सांगा.", "History"),
            Turn("पुनरुत्थान (Renaissance) काय होते आणि परिणाम काय?", "Renaissance"),
            Turn("सगळ्या विषयांचा 4 मुद्द्यांत सारांश द्या.", "Summary"),
            Turn("या विषयांवर 3 सराव प्रश्न द्या.", "Practice questions"),
            Turn("लघु निष्कर्ष लिहा.", "Conclusion"),
            Turn("इयत्ता 12 साठी कठीणपणा स्तर सांगा.", "Difficulty"),
            Turn("रिव्हिजनसाठी 3 टिप्स द्या.", "Revision tips"),
            Turn("आता झटपट पुनरावलोकन करा.", "Quick recap"),
        ],
    },
]


def get_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(enc, text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def call_chat_api(turn: Turn, session_id: str, user_id: str, language: str) -> Dict[str, Any]:
    payload = {
        "user_session_id": session_id,
        "user_id": user_id,
        "user_type": "student",
        "query": turn.query,
        "student_grade": STUDENT_GRADE,
        "language": language,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req) as response:
        duration = time.perf_counter() - start
        resp_data = json.loads(response.read().decode("utf-8"))
        resp_data["_latency_s"] = duration
        return resp_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enc = get_encoding(MODEL_NAME)
    run_start_ts = time.time()

    all_reports = []

    for convo in CONVERSATIONS:
        user_id = convo.get("user_id", DEFAULT_USER_ID)
        session_id = convo.get("session_id", DEFAULT_SESSION_ID)
        language = convo.get("language", "en")
        turns = convo.get("turns", [])
        log_totals_iter = None

        results = []
        totals = {"user_tokens": 0, "assistant_tokens": 0, "total_tokens": 0}

        print(f"Session: {session_id} | User: {user_id} | Lang: {language}")
        print(f"Model: {MODEL_NAME} | Turns: {len(turns)}")
        print("-" * 60)

        for idx, turn in enumerate(turns, 1):
            try:
                resp = call_chat_api(turn, session_id, user_id, language)
                message = resp.get("message", "")
            except urllib.error.HTTPError as e:
                print(f"HTTP error on turn {idx}: {e.code} {e.read().decode()}")
                break
            except Exception as e:
                print(f"Request failed on turn {idx}: {e}")
                break

            user_tokens = count_tokens(enc, turn.query)
            assistant_tokens = count_tokens(enc, message)
            total_tokens = user_tokens + assistant_tokens

            totals["user_tokens"] += user_tokens
            totals["assistant_tokens"] += assistant_tokens
            totals["total_tokens"] += total_tokens

            api_total_effective = (
                resp.get("total_tokens_with_background")
                if resp.get("total_tokens_with_background") is not None
                else resp.get("total_tokens", 0)
            )
            if api_total_effective == 0 and DEBUG_RESPONSE:
                print(f"[DEBUG] Response keys: {sorted(resp.keys())}")

            results.append({
                "turn": idx,
                "language": language,
                "description": turn.description,
                "query": turn.query,
                "response_preview": message[:200],
                "latency_s": resp.get("_latency_s", 0),
                "api_input_tokens": resp.get("input_tokens", 0),
                "api_output_tokens": resp.get("output_tokens", 0),
                "api_total_tokens": resp.get("total_tokens", 0),
                "api_bg_input_tokens": resp.get("background_input_tokens", 0),
                "api_bg_output_tokens": resp.get("background_output_tokens", 0),
                "api_bg_total_tokens": resp.get("background_total_tokens", 0),
                "api_total_with_background": resp.get("total_tokens_with_background", 0),
                "api_total_effective": api_total_effective or 0,
                "user_tokens": user_tokens,
                "assistant_tokens": assistant_tokens,
                "total_tokens": total_tokens,
            })

            print(
                f"{idx:02d} [{language}] api_total_effective={api_total_effective} "
                f"(api_in={resp.get('input_tokens', 0)} api_out={resp.get('output_tokens', 0)} bg={resp.get('background_total_tokens', 0)})"
            )

        report = {
            "session_id": session_id,
            "user_id": user_id,
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "turns": len(results),
            "totals": totals,
            "api_totals": {
                "input_tokens": sum(r["api_input_tokens"] for r in results),
                "output_tokens": sum(r["api_output_tokens"] for r in results),
                "total_tokens": sum(r["api_total_tokens"] for r in results),
                "total_effective": sum(r["api_total_effective"] for r in results),
            },
            "api_background_totals": {
                "input_tokens": sum(r["api_bg_input_tokens"] for r in results),
                "output_tokens": sum(r["api_bg_output_tokens"] for r in results),
                "total_tokens": sum(r["api_bg_total_tokens"] for r in results),
                "total_with_background": sum(r["api_total_with_background"] for r in results),
            },
            "log_totals": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "notes": "Use API totals for billing. Approximate counts are user+assistant text only. If LOG_FILE is set, log totals are parsed from [TOKEN_USAGE] TOTAL REQUEST CYCLE.",
            "results": results,
        }

        if LOG_FILE:
            log_entries = _parse_log_totals(LOG_FILE, run_start_ts)
            if log_entries:
                # Take the last N entries, aligned to turn count
                recent = log_entries[-len(results):]
                for i, r in enumerate(results):
                    if i < len(recent):
                        r["log_input_tokens"] = recent[i]["input_tokens"]
                        r["log_output_tokens"] = recent[i]["output_tokens"]
                        r["log_total_tokens"] = recent[i]["total_tokens"]
                        r["log_delta_total"] = r["log_total_tokens"] - r["api_total_effective"]
                    else:
                        r["log_input_tokens"] = 0
                        r["log_output_tokens"] = 0
                        r["log_total_tokens"] = 0
                        r["log_delta_total"] = 0
                report["log_totals"]["input_tokens"] = sum(r.get("log_input_tokens", 0) for r in results)
                report["log_totals"]["output_tokens"] = sum(r.get("log_output_tokens", 0) for r in results)
                report["log_totals"]["total_tokens"] = sum(r.get("log_total_tokens", 0) for r in results)
                report["log_totals"]["delta_total"] = sum(r.get("log_delta_total", 0) for r in results)
        all_reports.append(report)

        json_path = os.path.join(OUTPUT_DIR, f"token_usage_report_{session_id}.json")
        md_path = os.path.join(OUTPUT_DIR, f"token_usage_report_{session_id}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Markdown summary
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Token Usage Report\n\n")
            f.write(f"- Session: `{session_id}`\n")
            f.write(f"- User: `{user_id}`\n")
            f.write(f"- Model: `{MODEL_NAME}`\n")
            f.write(f"- Base URL: `{BASE_URL}`\n")
            f.write(f"- Turns: `{len(results)}`\n")
            f.write(f"- Total user tokens: `{totals['user_tokens']}`\n")
            f.write(f"- Total assistant tokens: `{totals['assistant_tokens']}`\n")
            f.write(f"- Total tokens: `{totals['total_tokens']}`\n")
            f.write(f"- API input tokens: `{report['api_totals']['input_tokens']}`\n")
            f.write(f"- API output tokens: `{report['api_totals']['output_tokens']}`\n")
            f.write(f"- API total tokens: `{report['api_totals']['total_tokens']}`\n")
            f.write(f"- API total effective: `{report['api_totals']['total_effective']}`\n")
            f.write(f"- API background input tokens: `{report['api_background_totals']['input_tokens']}`\n")
            f.write(f"- API background output tokens: `{report['api_background_totals']['output_tokens']}`\n")
            f.write(f"- API background total tokens: `{report['api_background_totals']['total_tokens']}`\n")
            f.write(f"- API total with background: `{report['api_background_totals']['total_with_background']}`\n")
            f.write(f"- Log input tokens: `{report['log_totals']['input_tokens']}`\n")
            f.write(f"- Log output tokens: `{report['log_totals']['output_tokens']}`\n")
            f.write(f"- Log total tokens: `{report['log_totals']['total_tokens']}`\n")
            if report["log_totals"].get("delta_total") is not None:
                f.write(f"- Log-API total delta: `{report['log_totals'].get('delta_total', 0)}`\n")
            f.write(f"- Note: Token counts are approximate (user+assistant text only).\n\n")
            f.write("| Turn | Lang | API Total Effective | API In | API Out | API Total | API BG In | API BG Out | API BG Total | API Total+BG | User Tokens | Assistant Tokens | Approx Total | Log In | Log Out | Log Total | Log-API Delta | Latency (s) | Description |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for r in results:
                f.write(
                    f"| {r['turn']} | {r['language']} | {r['api_total_effective']} | "
                    f"{r['api_input_tokens']} | {r['api_output_tokens']} | {r['api_total_tokens']} | "
                    f"{r['api_bg_input_tokens']} | {r['api_bg_output_tokens']} | {r['api_bg_total_tokens']} | {r['api_total_with_background']} | "
                    f"{r['user_tokens']} | {r['assistant_tokens']} | {r['total_tokens']} | "
                    f"{r.get('log_input_tokens', 0)} | {r.get('log_output_tokens', 0)} | {r.get('log_total_tokens', 0)} | "
                    f"{r.get('log_delta_total', 0)} | {r['latency_s']:.2f} | {r['description']} |\n"
                )

        print("-" * 60)
        print(f"Report written:")
        print(f"- {json_path}")
        print(f"- {md_path}")

    combined_path = os.path.join(OUTPUT_DIR, "token_usage_report_combined.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"Combined report: {combined_path}")


if __name__ == "__main__":
    main()
