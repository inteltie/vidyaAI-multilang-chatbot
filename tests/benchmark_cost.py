"""
VidyaAI Token Usage Benchmark
==============================

Runs two full multi-turn conversations against the deployed /chat API:
  - Case 1: English-only (no translation overhead)
  - Case 2: Hindi responses (translation node is triggered every turn)

For each turn it records:
  - Input / Output / Total tokens (from the response body)
  - Background tokens (summarization fired in background)
  - LLM call count
  - Response latency
  - First 120 chars of response (quality spot-check)

After all turns it prints a per-turn table and projects monthly costs.

Usage:
  uv run python tests/benchmark_cost.py

Requirements: httpx (already in pyproject.toml)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_URL = "https://multilang-chatbot.vidyaai.ai"
CHAT_URL = f"{BASE_URL}/chat"
HEALTH_URL = f"{BASE_URL}/health"

USER_ID = "benchmark-test-001"
USER_TYPE = "student"
STUDENT_GRADE = "B"

# 2026 OpenAI Pricing (per 1M tokens)
GPT4O_MINI_INPUT_PRICE   = 0.15   # USD per 1M input tokens
GPT4O_MINI_OUTPUT_PRICE  = 0.60   # USD per 1M output tokens
EMBEDDING_PRICE          = 0.13   # USD per 1M embedding tokens (text-embedding-3-large)
AVG_EMBEDDING_TOKENS     = 20     # Rough avg tokens per RAG query

# Timeout per request (seconds) — long because heavy queries can take time
REQUEST_TIMEOUT = 90.0

# HTTP client settings
HEADERS = {"Content-Type": "application/json"}


# ─────────────────────────────────────────────
# 15-TURN CONVERSATION SCRIPT
# ─────────────────────────────────────────────
# Case 1: English queries — no translation triggered
CONVERSATION_TURNS_EN: List[str] = [
    # Biology — foundational
    "What is photosynthesis and how does it work?",

    # Biology — follow-up (tests memory/context continuity)
    "What role does chlorophyll play in this process? And what happens if a plant lacks sunlight?",

    # Physics — new subject (tests topic shift handling)
    "Can you explain Newton's three laws of motion with examples?",

    # Physics — follow-up + application (tests deep reasoning)
    "How does Newton's second law relate to a car that suddenly brakes? Include a force calculation example.",

    # Cross-domain (tests concept linking)
    "What is the fundamental difference between a cell and an atom? Which is smaller?",

    # Geography — detailed process
    "Explain the water cycle in detail, including evaporation, condensation, and precipitation.",

    # History — complex cause-and-effect
    "What was the French Revolution? What were the main causes that led to it?",

    # History — follow-up (tests continuity across complex topic)
    "How did the revolution change the government and social structure of France?",

    # Math — procedural
    "What is a quadratic equation and what methods can I use to solve it?",

    # Math — step-by-step (tests structured output quality)
    "Solve 2x\u00b2 + 7x + 3 = 0 step by step using the quadratic formula.",

    # Biology — advanced
    "What is DNA replication? Describe the key enzymes involved.",

    # Biology — complex comparison (tests nuanced differentiation)
    "How does mitosis differ from meiosis? When does each process occur in the human body?",

    # Chemistry — conceptual
    "What is the difference between acids and bases? Give examples of each from daily life.",

    # Memory / meta-cognition (tests conversation summary capability)
    "Can you give me a brief summary of all the topics we have discussed in this conversation?",

    # Reflection / off-topic handling
    "Which of the topics we covered today do you think is the most important for a student to master first?",
]

# Case 2: HINDI queries — language detector must identify Hindi for translation to fire.
# The translation node reads state['language'] which is set from the DETECTED query language,
# NOT from the request payload's 'language' field.
CONVERSATION_TURNS_HI: List[str] = [
    # Biology — foundational
    "\u092a\u094d\u0930\u0915\u093e\u0936 \u0938\u0902\u0936\u094d\u0932\u0947\u0937\u0923 \u0915\u094d\u092f\u093e \u0939\u0948 \u0914\u0930 \u092f\u0939 \u0915\u0948\u0938\u0947 \u0915\u093e\u092e \u0915\u0930\u0924\u093e \u0939\u0948?",

    # Biology — follow-up
    "\u0907\u0938 \u092a\u094d\u0930\u0915\u094d\u0930\u093f\u092f\u093e \u092e\u0947\u0902 \u0915\u094d\u0932\u094b\u0930\u094b\u092b\u093f\u0932 \u0915\u094d\u092f\u093e \u092d\u0942\u092e\u093f\u0915\u093e \u0928\u093f\u092d\u093e\u0924\u093e \u0939\u0948? \u0914\u0930 \u0905\u0917\u0930 \u092a\u094c\u0927\u0947 \u0915\u094b \u0938\u0942\u0930\u092f \u0915\u0940 \u0930\u094b\u0936\u0928\u0940 \u0928\u0939\u0940\u0902 \u092e\u093f\u0932\u0947 \u0924\u094b \u0915\u094d\u092f\u093e \u0939\u094b\u0924\u093e \u0939\u0948?",

    # Physics — new subject
    "\u0928\u094d\u092f\u0942\u091f\u0928 \u0915\u0947 \u0917\u0924\u093f \u0915\u0947 \u0924\u0940\u0928 \u0928\u093f\u092f\u092e \u0909\u0926\u093e\u0939\u0930\u0923 \u0938\u0939\u093f\u0924 \u0938\u092e\u091d\u093e\u0907\u090f?",

    # Physics — follow-up
    "\u0928\u094d\u092f\u0942\u091f\u0928 \u0915\u093e \u0926\u0942\u0938\u0930\u093e \u0928\u093f\u092f\u092e \u090f\u0915 \u0905\u091a\u093e\u0928\u0915 \u092c\u094d\u0930\u0947\u0915 \u0932\u0917\u093e\u0928\u0947 \u0935\u093e\u0932\u0940 \u0915\u093e\u0930 \u0938\u0947 \u0915\u0948\u0938\u0947 \u0938\u0902\u092c\u0902\u0927\u093f\u0924 \u0939\u0948? \u092c\u0932 \u0915\u0940 \u0917\u0923\u0928\u093e \u0915\u093e \u0909\u0926\u093e\u0939\u0930\u0923 \u0926\u0947\u0902?",

    # Cross-domain
    "\u0915\u094b\u0936\u093f\u0915\u093e \u0914\u0930 \u092a\u0930\u092e\u093e\u0923\u0941 \u092e\u0947\u0902 \u092e\u0942\u0932 \u0905\u0902\u0924\u0930 \u0915\u094d\u092f\u093e \u0939\u0948? \u0915\u094c\u0928 \u0938\u093e \u091b\u094b\u091f\u093e \u0939\u0948?",

    # Geography
    "\u091c\u0932 \u091a\u0915\u094d\u0930 \u0915\u094b \u0935\u093e\u0937\u094d\u092a\u0940\u0915\u0930\u0923, \u0938\u0902\u0918\u0928\u0928 \u0914\u0930 \u0935\u0930\u094d\u0937\u093e \u0938\u0939\u093f\u0924 \u0935\u093f\u0938\u094d\u0924\u093e\u0930 \u0938\u0947 \u0938\u092e\u091d\u093e\u0907\u090f?",

    # History
    "\u092b\u094d\u0930\u093e\u0902\u0938\u0940\u0938\u0940 \u0915\u094d\u0930\u093e\u0902\u0924\u093f \u0915\u094d\u092f\u093e \u0925\u0940? \u0907\u0938\u0915\u0947 \u092e\u0941\u0916\u094d\u092f \u0915\u093e\u0930\u0923 \u0915\u094d\u092f\u093e \u0925\u0947?",

    # History — follow-up
    "\u0907\u0938 \u0915\u094d\u0930\u093e\u0902\u0924\u093f \u0928\u0947 \u092b\u094d\u0930\u093e\u0902\u0938 \u0915\u0940 \u0938\u0930\u0915\u093e\u0930 \u0914\u0930 \u0938\u093e\u092e\u093e\u091c\u093f\u0915 \u0922\u093e\u0902\u091a\u0947 \u0915\u094b \u0915\u0948\u0938\u0947 \u092c\u0926\u0932\u093e?",

    # Math
    "\u0926\u094d\u0935\u093f\u0918\u093e\u0924 \u0938\u092e\u0940\u0915\u0930\u0923 \u0915\u094d\u092f\u093e \u0939\u0948 \u0914\u0930 \u092e\u0948\u0902 \u0907\u0938\u0947 \u0939\u0932 \u0915\u0930\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0915\u094b\u0928 \u0938\u0940 \u0935\u093f\u0927\u093f\u092f\u093e\u0902 \u0909\u092a\u092f\u094b\u0917 \u0915\u0930 \u0938\u0915\u0924\u093e \u0939\u0942\u0902?",

    # Math — step-by-step
    "2x\u00b2 + 7x + 3 = 0 \u0915\u094b \u0926\u094d\u0935\u093f\u0918\u093e\u0924 \u0938\u0942\u0924\u094d\u0930 \u0915\u093e \u0909\u092a\u092f\u094b\u0917 \u0915\u0930\u0915\u0947 \u091a\u0930\u0923-\u0926\u0930-\u091a\u0930\u0923 \u0939\u0932 \u0915\u0930\u0947\u0902?",

    # Biology — advanced
    "DNA \u092a\u094d\u0930\u0924\u093f\u0915\u0943\u0924\u093f \u0915\u094d\u092f\u093e \u0939\u0948? \u0907\u0938\u092e\u0947\u0902 \u0936\u093e\u092e\u093f\u0932 \u092e\u0941\u0916\u094d\u092f \u090f\u0902\u091c\u093e\u0907\u092e\u094b\u0902 \u0915\u093e \u0935\u0930\u094d\u0923\u0928 \u0915\u0930\u0947\u0902?",

    # Biology — complex comparison
    "\u0915\u094b\u0936\u093f\u0915\u093e \u0935\u093f\u092d\u093e\u091c\u0928 \u0914\u0930 \u0905\u0930\u094d\u0927\u0938\u0942\u0924\u094d\u0930\u0940 \u0935\u093f\u092d\u093e\u091c\u0928 \u092e\u0947\u0902 \u0905\u0902\u0924\u0930 \u0915\u094d\u092f\u093e \u0939\u0948? \u092e\u093e\u0928\u0935 \u0936\u0930\u0940\u0930 \u092e\u0947\u0902 \u092f\u0947 \u0915\u092c \u0939\u094b\u0924\u0947 \u0939\u0948\u0902?",

    # Chemistry
    "\u0905\u092e\u094d\u0932 \u0914\u0930 \u0915\u094d\u0937\u093e\u0930 \u092e\u0947\u0902 \u0905\u0902\u0924\u0930 \u0915\u094d\u092f\u093e \u0939\u0948? \u0926\u0948\u0928\u093f\u0915 \u091c\u0940\u0935\u0928 \u0938\u0947 \u0909\u0926\u093e\u0939\u0930\u0923 \u0926\u0947\u0902?",

    # Memory / meta-cognition
    "\u0907\u0938 \u092c\u093e\u0924\u091a\u0940\u0924 \u092e\u0947\u0902 \u0939\u092e\u0928\u0947 \u091c\u094b \u0935\u093f\u0937\u092f \u091a\u0930\u094d\u091a\u093e \u0915\u093f\u090f \u0909\u0928\u0915\u093e \u0938\u0902\u0915\u094d\u0937\u093f\u092a\u094d\u0924 \u0938\u093e\u0930\u093e\u0902\u0936 \u0926\u0947 \u0938\u0915\u0924\u0947 \u0939\u0948\u0902?",

    # Reflection
    "\u0906\u092a\u0915\u0947 \u0935\u093f\u091a\u093e\u0930 \u0938\u0947 \u0906\u091c \u0939\u092e\u0928\u0947 \u091c\u094b \u0935\u093f\u0937\u092f \u092a\u0922\u093c\u0947, \u0909\u0928\u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u091b\u093e\u0924\u094d\u0930 \u0915\u0947 \u0932\u093f\u090f \u092a\u0939\u0932\u0947 \u0938\u092c\u0938\u0947 \u092e\u0939\u0924\u094d\u0935\u092a\u0942\u0930\u094d\u0923 \u0915\u094c\u0928 \u0938\u093e \u0939\u0948?",
]


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class TurnResult:
    turn: int
    query: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    bg_input_tokens: int = 0
    bg_output_tokens: int = 0
    bg_total_tokens: int = 0
    total_with_bg: int = 0
    llm_calls: int = 0
    intent: str = ""
    language: str = ""
    latency_ms: float = 0.0
    response_preview: str = ""
    error: Optional[str] = None
    embedding_tokens: int = AVG_EMBEDDING_TOKENS  # estimated per turn

    @property
    def grand_total_tokens(self) -> int:
        return self.total_with_bg  # includes background summarisation

    @property
    def input_cost(self) -> float:
        return self.input_tokens * GPT4O_MINI_INPUT_PRICE / 1_000_000

    @property
    def output_cost(self) -> float:
        return self.output_tokens * GPT4O_MINI_OUTPUT_PRICE / 1_000_000

    @property
    def bg_input_cost(self) -> float:
        return self.bg_input_tokens * GPT4O_MINI_INPUT_PRICE / 1_000_000

    @property
    def bg_output_cost(self) -> float:
        return self.bg_output_tokens * GPT4O_MINI_OUTPUT_PRICE / 1_000_000

    @property
    def embedding_cost(self) -> float:
        return self.embedding_tokens * EMBEDDING_PRICE / 1_000_000

    @property
    def total_cost(self) -> float:
        return (
            self.input_cost
            + self.output_cost
            + self.bg_input_cost
            + self.bg_output_cost
            + self.embedding_cost
        )


@dataclass
class CaseResult:
    name: str
    language: str
    turns: List[TurnResult] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"benchmark-{uuid.uuid4().hex[:8]}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def divider(char: str = "─", width: int = 100) -> str:
    return char * width


def clamp_str(s: str, n: int = 120) -> str:
    return (s[:n] + "…") if len(s) > n else s


def compute_cost_str(tokens: int, price_per_million: float) -> str:
    return f"${tokens * price_per_million / 1_000_000:.6f}"


# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────
async def check_health(client: httpx.AsyncClient) -> bool:
    print(f"  Checking server health at {HEALTH_URL} …")
    try:
        r = await client.get(HEALTH_URL, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        print(f"  ✓ Server is UP — {data}")
        return True
    except Exception as exc:
        print(f"  ✗ Server health check FAILED: {exc}")
        return False


async def send_chat(
    client: httpx.AsyncClient,
    session_id: str,
    query: str,
    language: str,
) -> Dict[str, Any]:
    payload = {
        "user_session_id": session_id,
        "user_id": USER_ID,
        "user_type": USER_TYPE,
        "query": query,
        "language": language,
        "agent_mode": "standard",
        "student_grade": STUDENT_GRADE,
    }
    t0 = time.perf_counter()
    r = await client.post(CHAT_URL, json=payload, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = latency_ms
    return data


# ─────────────────────────────────────────────
# CORE TEST RUNNER
# ─────────────────────────────────────────────
async def run_case(
    client: httpx.AsyncClient,
    case: CaseResult,
    turns: List[str],
) -> CaseResult:
    print(f"\n{'═' * 100}")
    print(f"  CASE: {case.name}  │  Language: {case.language.upper()}  │  Session: {case.session_id}")
    print(f"{'═' * 100}\n")

    for i, query in enumerate(turns, start=1):
        print(f"  Turn {i:02d}/{len(turns)}: {clamp_str(query, 80)}")

        result = TurnResult(turn=i, query=query)

        try:
            data = await send_chat(client, case.session_id, query, case.language)
            result.input_tokens    = data.get("input_tokens", 0)
            result.output_tokens   = data.get("output_tokens", 0)
            result.total_tokens    = data.get("total_tokens", 0)
            result.bg_input_tokens = data.get("background_input_tokens", 0)
            result.bg_output_tokens= data.get("background_output_tokens", 0)
            result.bg_total_tokens = data.get("background_total_tokens", 0)
            result.total_with_bg   = data.get("total_tokens_with_background", result.total_tokens)
            result.llm_calls       = data.get("llm_calls", 0)
            result.intent          = data.get("intent", "")
            result.language        = data.get("language", case.language)
            result.latency_ms      = data.get("_latency_ms", 0.0)
            result.response_preview= clamp_str(data.get("message", ""), 120)

            print(
                f"         ✓  in={result.input_tokens:,} out={result.output_tokens:,} "
                f"bg={result.bg_total_tokens:,}  total={result.total_with_bg:,}  "
                f"latency={result.latency_ms/1000:.1f}s  cost=${result.total_cost:.6f}"
            )
            print(f"         Preview: {result.response_preview}\n")

        except httpx.HTTPStatusError as exc:
            result.error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            print(f"         ✗ HTTP error: {result.error}\n")
        except Exception as exc:
            result.error = str(exc)
            print(f"         ✗ Error: {result.error}\n")

        case.turns.append(result)

        # Small pause to avoid rate limiting / hammering the server
        await asyncio.sleep(1.5)

    return case


# ─────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────
def print_case_report(case: CaseResult) -> None:
    success = [t for t in case.turns if t.error is None]
    failed  = [t for t in case.turns if t.error is not None]

    total_input    = sum(t.input_tokens    for t in success)
    total_output   = sum(t.output_tokens   for t in success)
    total_bg_in    = sum(t.bg_input_tokens for t in success)
    total_bg_out   = sum(t.bg_output_tokens for t in success)
    total_embed    = sum(t.embedding_tokens for t in success)
    grand_total    = sum(t.total_with_bg   for t in success) + total_embed
    total_turns    = len(success)
    avg_latency    = (sum(t.latency_ms for t in success) / total_turns) if total_turns else 0

    cost_input   = total_input    * GPT4O_MINI_INPUT_PRICE  / 1_000_000
    cost_output  = total_output   * GPT4O_MINI_OUTPUT_PRICE / 1_000_000
    cost_bg_in   = total_bg_in   * GPT4O_MINI_INPUT_PRICE  / 1_000_000
    cost_bg_out  = total_bg_out  * GPT4O_MINI_OUTPUT_PRICE / 1_000_000
    cost_embed   = total_embed   * EMBEDDING_PRICE          / 1_000_000
    total_cost   = cost_input + cost_output + cost_bg_in + cost_bg_out + cost_embed

    col = {
        "turn":     5, "query": 44, "in": 8, "out": 8, "bg": 7,
        "total": 8, "llm": 4, "lat": 7, "cost": 10, "quality": 20,
    }

    header = (
        f"{'#':>{col['turn']}} │ {'Query':<{col['query']}} │ "
        f"{'Input':>{col['in']}} │ {'Output':>{col['out']}} │ {'BG':>{col['bg']}} │ "
        f"{'Total':>{col['total']}} │ {'LLM':>{col['llm']}} │ "
        f"{'Lat(s)':>{col['lat']}} │ {'Cost($)':>{col['cost']}} │ Response Preview"
    )

    print(f"\n{divider('═')}")
    print(f"  RESULTS: {case.name}  │  Language: {case.language.upper()}")
    print(divider('─'))
    print(header)
    print(divider('─'))

    for t in case.turns:
        if t.error:
            print(
                f"  {t.turn:>{col['turn']}} │ {clamp_str(t.query, col['query']):<{col['query']}} │ "
                f"{'ERROR':<{col['in'] + col['out'] + col['bg'] + col['total'] + 12}} │ {t.error[:40]}"
            )
        else:
            print(
                f"  {t.turn:>{col['turn']}} │ {clamp_str(t.query, col['query']):<{col['query']}} │ "
                f"  {t.input_tokens:>{col['in']},} │ {t.output_tokens:>{col['out']},} │ "
                f"{t.bg_total_tokens:>{col['bg']},} │ {t.total_with_bg:>{col['total']},} │ "
                f"  {t.llm_calls:>{col['llm']}} │ {t.latency_ms/1000:>{col['lat']}.1f} │ "
                f"  {t.total_cost:>{col['cost']}.6f} │ {clamp_str(t.response_preview, 45)}"
            )

    print(divider('─'))
    print(f"\n  TOKEN SUMMARY")
    print(f"  {'Main Input Tokens:':<35} {total_input:>10,}   → ${cost_input:.6f}")
    print(f"  {'Main Output Tokens:':<35} {total_output:>10,}   → ${cost_output:.6f}")
    print(f"  {'Background Input Tokens:':<35} {total_bg_in:>10,}   → ${cost_bg_in:.6f}")
    print(f"  {'Background Output Tokens:':<35} {total_bg_out:>10,}   → ${cost_bg_out:.6f}")
    print(f"  {'Estimated Embedding Tokens:':<35} {total_embed:>10,}   → ${cost_embed:.6f}")
    print(divider('─'))
    print(f"  {'TOTAL TOKENS (incl. BG + embed):':<35} {grand_total:>10,}")
    print(f"  {'TOTAL COST (this session):':<35} ${total_cost:.6f}")
    print(f"  {'Avg Latency per Turn:':<35} {avg_latency/1000:.2f}s")
    print(f"  {'Failed Turns:':<35} {len(failed)}")
    print(divider('═'))

    return total_cost


def print_monthly_projection(case_results: List[tuple]) -> None:
    SESSIONS_PER_DAY   = 1_000
    DAYS_PER_MONTH     = 30
    SESSIONS_PER_MONTH = SESSIONS_PER_DAY * DAYS_PER_MONTH

    print(f"\n{'═' * 100}")
    print(f"  MONTHLY COST PROJECTION  (assuming {SESSIONS_PER_DAY:,} sessions/day, {DAYS_PER_MONTH} days)")
    print(divider('─'))
    print(f"  {'Case':<40} {'Cost/Session':>14} {'Daily Cost':>14} {'Monthly Cost':>14}")
    print(divider('─'))

    for name, cost in case_results:
        daily   = cost * SESSIONS_PER_DAY
        monthly = cost * SESSIONS_PER_MONTH
        print(f"  {name:<40} ${cost:>13.5f} ${daily:>13.2f}  ${monthly:>12.2f}")

    print(divider('═'))

    total_saves_pct = 0
    if len(case_results) >= 2:
        base  = case_results[0][1]
        extra = case_results[1][1]
        if base > 0:
            increase_pct = ((extra - base) / base) * 100
            print(
                f"\n  Translation overhead adds {increase_pct:+.1f}% cost per session compared to English-only.\n"
            )


def save_json_report(cases: List[CaseResult], filename: str) -> None:
    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_url": BASE_URL,
        "pricing": {
            "gpt4o_mini_input_per_1m": GPT4O_MINI_INPUT_PRICE,
            "gpt4o_mini_output_per_1m": GPT4O_MINI_OUTPUT_PRICE,
            "embedding_per_1m": EMBEDDING_PRICE,
        },
        "cases": [],
    }

    for case in cases:
        case_data: Dict[str, Any] = {
            "name": case.name,
            "language": case.language,
            "session_id": case.session_id,
            "turns": [],
        }
        for t in case.turns:
            turn_data = {
                "turn": t.turn,
                "query": t.query,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "total_tokens": t.total_tokens,
                "bg_input_tokens": t.bg_input_tokens,
                "bg_output_tokens": t.bg_output_tokens,
                "bg_total_tokens": t.bg_total_tokens,
                "total_with_bg": t.total_with_bg,
                "embedding_tokens": t.embedding_tokens,
                "llm_calls": t.llm_calls,
                "intent": t.intent,
                "language": t.language,
                "latency_ms": round(t.latency_ms, 2),
                "total_cost_usd": round(t.total_cost, 8),
                "response_preview": t.response_preview,
                "error": t.error,
            }
            case_data["turns"].append(turn_data)
        report["cases"].append(case_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Full JSON report saved → {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def main() -> None:
    print()
    print("╔" + "═" * 98 + "╗")
    print("║" + "  VidyaAI Token Usage Benchmark — April 2026".center(98) + "║")
    print("║" + f"  Target: {CHAT_URL}".center(98) + "║")
    print("║" + f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(98) + "║")
    print("╚" + "═" * 98 + "╝")

    async with httpx.AsyncClient() as client:
        # ── Health check ──────────────────────────────────────────────────────────
        print(f"\n{divider()}")
        print("  STEP 1: Server Health Check")
        print(divider())
        alive = await check_health(client)
        if not alive:
            print("\n  ✗ Cannot proceed — server is not reachable. Aborting.")
            sys.exit(1)

        # ── Case 1: English Only ──────────────────────────────────────────────────
        case1 = CaseResult(
            name="Case 1 — English Only",
            language="en",
        )
        await run_case(client, case1, CONVERSATION_TURNS_EN)

        # ── Case 2: Hindi Translation ─────────────────────────────────────────────
        # IMPORTANT: Queries are sent IN HINDI so the fastText language detector
        # correctly identifies the language as 'hi' and triggers the translation node.
        # Sending English text with language='hi' bypasses translation entirely because
        # the detector overrides the payload's language field.
        case2 = CaseResult(
            name="Case 2 — Hindi Translation (Actual Hindi Queries)",
            language="hi",
        )
        await run_case(client, case2, CONVERSATION_TURNS_HI)

    # ── Print Full Reports ────────────────────────────────────────────────────
    cost1 = print_case_report(case1)
    cost2 = print_case_report(case2)

    # ── Monthly Projection ────────────────────────────────────────────────────
    print_monthly_projection(
        [
            (case1.name, cost1),
            (case2.name, cost2),
        ]
    )

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_file = f"tests/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json_report([case1, case2], out_file)

    print("\n  ✓ Benchmark complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
