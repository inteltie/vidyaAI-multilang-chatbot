"""
VidyaAI Token Usage Benchmark
==============================

Runs multi-turn conversations against the deployed /chat API:
  - Case 1: Standard mode (agent_mode=standard)
  - Case 2: Interactive mode (agent_mode=interactive)
  - Case 3: Interactive + image upload (vision costing)

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

# Case 3: interactive + image (3 turns)
MINIMAL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
# Case 4: interactive image generation (Google Gemini; requires GEMINI_API_KEY)
CASE4_TURNS: List[Dict[str, Any]] = [
    {"query": "Draw a simple photosynthesis diagram with labeled steps.", "images": []},
]

CASE3_TURNS: List[Dict[str, Any]] = [
    {"query": "What is photosynthesis in one sentence?", "images": []},
    {
        "query": "What do you see in this diagram? Describe it briefly.",
        "images": [{"mime_type": "image/png", "data": MINIMAL_PNG_BASE64}],
    },
    {"query": "What did you see in the image from the previous step?", "images": []},
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
    embedding_tokens: int = 0
    cost_usd_llm: float = 0.0
    cost_usd_web_search: float = 0.0
    cost_usd_vision: float = 0.0
    cost_usd_background: float = 0.0
    cost_usd_embedding: float = 0.0
    cost_usd_total: float = 0.0
    cost_by_component_usd: Dict[str, float] = field(default_factory=dict)
    web_search_calls: int = 0
    images_analyzed: int = 0
    cost_usd_image_generation: float = 0.0
    image_generation_calls: int = 0
    image_generation_backend: str = ""
    image_model_used: str = ""
    generated_images_count: int = 0

    @property
    def grand_total_tokens(self) -> int:
        return self.total_with_bg

    @property
    def total_cost(self) -> float:
        """Prefer server-computed cost_usd_total when available."""
        if self.cost_usd_total > 0:
            return self.cost_usd_total
        return (
            self.input_tokens * GPT4O_MINI_INPUT_PRICE / 1_000_000
            + self.output_tokens * GPT4O_MINI_OUTPUT_PRICE / 1_000_000
            + self.bg_input_tokens * GPT4O_MINI_INPUT_PRICE / 1_000_000
            + self.bg_output_tokens * GPT4O_MINI_OUTPUT_PRICE / 1_000_000
            + self.embedding_tokens * EMBEDDING_PRICE / 1_000_000
        )


@dataclass
class CaseResult:
    name: str
    language: str
    agent_mode: str = "standard"
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
    agent_mode: str,
    images: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    payload = {
        "user_session_id": session_id,
        "user_id": USER_ID,
        "user_type": USER_TYPE,
        "query": query,
        "language": language,
        "agent_mode": agent_mode,
        "student_grade": STUDENT_GRADE,
    }
    if images:
        payload["images"] = images
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
    turns: List[str] | List[Dict[str, Any]],
) -> CaseResult:
    print(f"\n{'═' * 100}")
    print(
        f"  CASE: {case.name}  │  Language: {case.language.upper()}  │  "
        f"Mode: {case.agent_mode}  │  Session: {case.session_id}"
    )
    print(f"{'═' * 100}\n")

    for i, turn in enumerate(turns, start=1):
        if isinstance(turn, dict):
            query = str(turn.get("query", ""))
            images = turn.get("images") or []
        else:
            query = str(turn)
            images = []

        print(f"  Turn {i:02d}/{len(turns)}: {clamp_str(query, 80)}")

        result = TurnResult(turn=i, query=query)

        try:
            data = await send_chat(
                client, case.session_id, query, case.language, case.agent_mode, images=images or None
            )
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
            result.embedding_tokens = data.get("embedding_tokens", 0)
            result.cost_usd_llm = float(data.get("cost_usd_llm", 0) or 0)
            result.cost_usd_web_search = float(data.get("cost_usd_web_search", 0) or 0)
            result.cost_usd_vision = float(data.get("cost_usd_vision", 0) or 0)
            result.cost_usd_background = float(data.get("cost_usd_background", 0) or 0)
            result.cost_usd_embedding = float(data.get("cost_usd_embedding", 0) or 0)
            result.cost_usd_total = float(data.get("cost_usd_total", 0) or 0)
            result.cost_by_component_usd = data.get("cost_by_component_usd") or {}
            result.web_search_calls = int(data.get("web_search_calls", 0) or 0)
            result.images_analyzed = int(data.get("images_analyzed", 0) or 0)
            result.cost_usd_image_generation = float(
                data.get("cost_usd_image_generation", 0) or 0
            )
            result.image_generation_calls = int(
                data.get("image_generation_calls", 0) or 0
            )
            result.image_generation_backend = str(
                data.get("image_generation_backend", "") or ""
            )
            result.image_model_used = str(data.get("image_model_used", "") or "")
            gen_imgs = data.get("generated_images") or []
            result.generated_images_count = len(gen_imgs)

            print(
                f"         ✓  in={result.input_tokens:,} out={result.output_tokens:,} "
                f"bg={result.bg_total_tokens:,}  total={result.total_with_bg:,}  "
                f"web=${result.cost_usd_web_search:.6f} vision=${result.cost_usd_vision:.6f} "
                f"gen=${result.cost_usd_image_generation:.6f} "
                f"web_calls={result.web_search_calls} imgs={result.images_analyzed} "
                f"gen_calls={result.image_generation_calls} backend={result.image_generation_backend} "
                f"gen_imgs={result.generated_images_count} "
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

    total_cost = sum(t.cost_usd_total for t in success) or sum(t.total_cost for t in success)
    total_web = sum(t.cost_usd_web_search for t in success)
    total_vision = sum(t.cost_usd_vision for t in success)
    total_web_calls = sum(t.web_search_calls for t in success)

    col = {
        "turn": 5, "query": 36, "in": 8, "out": 8, "web": 8, "vis": 8,
        "cost": 10, "lat": 7, "llm": 4,
    }

    header = (
        f"{'#':>{col['turn']}} │ {'Query':<{col['query']}} │ "
        f"{'Input':>{col['in']}} │ {'Output':>{col['out']}} │ "
        f"{'Web$':>{col['web']}} │ {'Vis$':>{col['vis']}} │ "
        f"{'Cost($)':>{col['cost']}} │ {'Lat(s)':>{col['lat']}} │ Response Preview"
    )

    print(f"\n{divider('═')}")
    print(f"  RESULTS: {case.name}  │  Language: {case.language.upper()}  │  Mode: {case.agent_mode}")
    print(divider('─'))
    print(header)
    print(divider('─'))

    for t in case.turns:
        if t.error:
            print(
                f"  {t.turn:>{col['turn']}} │ {clamp_str(t.query, col['query']):<{col['query']}} │ "
                f"{'ERROR':<50} │ {t.error[:40]}"
            )
        else:
            print(
                f"  {t.turn:>{col['turn']}} │ {clamp_str(t.query, col['query']):<{col['query']}} │ "
                f"  {t.input_tokens:>{col['in']},} │ {t.output_tokens:>{col['out']},} │ "
                f"  {t.cost_usd_web_search:>{col['web']}.6f} │ {t.cost_usd_vision:>{col['vis']}.6f} │ "
                f"  {t.total_cost:>{col['cost']}.6f} │ {t.latency_ms/1000:>{col['lat']}.1f} │ "
                f"{clamp_str(t.response_preview, 40)}"
            )

    print(divider('─'))
    print(f"\n  TOKEN SUMMARY")
    print(f"  {'Main Input Tokens:':<35} {total_input:>10,}")
    print(f"  {'Main Output Tokens:':<35} {total_output:>10,}")
    print(f"  {'Background Tokens:':<35} {total_bg_in + total_bg_out:>10,}")
    print(f"  {'Embedding Tokens (API):':<35} {total_embed:>10,}")
    print(divider('─'))
    print(f"  {'TOTAL TOKENS (incl. BG):':<35} {grand_total:>10,}")
    print(f"  {'Web search cost (API):':<35} ${total_web:.6f}  ({total_web_calls} calls)")
    print(f"  {'Vision cost (API):':<35} ${total_vision:.6f}")
    print(f"  {'TOTAL COST (API, this session):':<35} ${total_cost:.6f}")
    if total_turns:
        print(f"  {'AVG COST PER QUERY:':<35} ${(total_cost / total_turns):.6f}")
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
            "agent_mode": case.agent_mode,
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
                "cost_usd_llm": t.cost_usd_llm,
                "cost_usd_web_search": t.cost_usd_web_search,
                "cost_usd_vision": t.cost_usd_vision,
                "cost_usd_background": t.cost_usd_background,
                "cost_usd_embedding": t.cost_usd_embedding,
                "cost_usd_total": t.cost_usd_total,
                "cost_by_component_usd": t.cost_by_component_usd,
                "web_search_calls": t.web_search_calls,
                "images_analyzed": t.images_analyzed,
                "cost_usd_image_generation": t.cost_usd_image_generation,
                "image_generation_calls": t.image_generation_calls,
                "image_generation_backend": t.image_generation_backend,
                "image_model_used": t.image_model_used,
                "generated_images_count": t.generated_images_count,
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

        # ── Case 1: Standard mode ─────────────────────────────────────────────────
        case1 = CaseResult(
            name="Case 1 — Standard Mode (English)",
            language="en",
            agent_mode="standard",
        )
        await run_case(client, case1, CONVERSATION_TURNS_EN)

        # ── Case 2: Interactive mode ──────────────────────────────────────────────
        case2 = CaseResult(
            name="Case 2 — Interactive Mode (English)",
            language="en",
            agent_mode="interactive",
        )
        await run_case(client, case2, CONVERSATION_TURNS_EN)

        # ── Case 3: Interactive + image ───────────────────────────────────────────
        case3 = CaseResult(
            name="Case 3 — Interactive + Image (English)",
            language="en",
            agent_mode="interactive",
        )
        await run_case(client, case3, CASE3_TURNS)

        # ── Case 4: Image generation ──────────────────────────────────────────────
        case4 = CaseResult(
            name="Case 4 — Image Generation (interactive)",
            language="en",
            agent_mode="interactive",
        )
        await run_case(client, case4, CASE4_TURNS)

    # ── Print Full Reports ────────────────────────────────────────────────────
    cost1 = print_case_report(case1)
    cost2 = print_case_report(case2)
    cost3 = print_case_report(case3)
    cost4 = print_case_report(case4)

    # ── Monthly Projection ────────────────────────────────────────────────────
    print_monthly_projection(
        [
            (case1.name, cost1),
            (case2.name, cost2),
            (case3.name, cost3),
            (case4.name, cost4),
        ]
    )

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_file = f"tests/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json_report([case1, case2, case3, case4], out_file)

    print("\n  ✓ Benchmark complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
