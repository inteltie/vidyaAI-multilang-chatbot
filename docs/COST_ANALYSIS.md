# VidyaAI Chatbot — Cost Analysis & Calculation Guide

> **Last Updated:** April 2026  
> **Based on:** Live benchmark run against `https://multilang-chatbot.vidyaai.ai`  
> **Benchmark file:** `tests/benchmark_results_20260406_131100.json`

---

## 1. Overview

VidyaAI uses **OpenAI's API** for all language model tasks. Costs are calculated on a **pay-per-token** basis — you pay for every token sent to (input) and received from (output) the model.

A **token** is roughly:
- ~¾ of an English word (e.g. "photosynthesis" = 1 token)
- ~1.5–3× more tokens for Devanagari/Hindi script per word
- Numbers, punctuation, and spaces each count as tokens too

---

## 2. Models Used & Their Pricing (April 2026)

| Role | Model | Input Price | Output Price |
|:---|:---|:---|:---|
| Main LLM (all agents) | `gpt-4o-mini` | **$0.15 / 1M tokens** | **$0.60 / 1M tokens** |
| Validator | `gpt-4o-mini` | $0.15 / 1M tokens | $0.60 / 1M tokens |
| Translator | `gpt-4o-mini` | $0.15 / 1M tokens | $0.60 / 1M tokens |
| Query Classifier | `gpt-4o-mini` | $0.15 / 1M tokens | $0.60 / 1M tokens |
| Embeddings (RAG) | `text-embedding-3-large` | **$0.13 / 1M tokens** | *(input only)* |

> Chat, validation, and translation use `gpt-4o-mini` token rates. Costs are reported **by component** in the API (`cost_by_component_usd`).

### Web search tool (`web_search_preview`)

| Item | Rate (non-reasoning, e.g. `gpt-4o-mini`) |
|:---|:---|
| Tool invocation | **$25 / 1,000 calls** (~$0.025 per billable call) |
| Search content tokens | **Free** (usage may still appear in logs; not double-billed) |

Cache hits do not increment `web_search_calls`. Configure via `WEB_SEARCH_FEE_PER_1K_CALLS_USD` and `WEB_SEARCH_CONTENT_TOKENS_BILLED`.

### Interactive images (vision)

| Item | Rate |
|:---|:---|
| `ImageAnalyzerAgent` (gpt-4o-mini vision) | $0.15 / 1M input, $0.60 / 1M output |
| `ImageContextGateAgent` (nano, ambiguous only) | $0.10 / 1M input, $0.40 / 1M output |

Reported separately as `cost_usd_vision` (analyzer) and included in `cost_usd_llm` for the nano gate.

**Vision detail:** Set `VISION_DETAIL=high` (default) for dense diagrams and small labels on `gpt-4o-mini`. Use `low` only for coarse scene description to save tokens.

### Interactive image generation (Google Gemini)

| Item | Value |
|:---|:---|
| Provider | Google Gemini API (`google-genai` SDK) |
| Model | `gemini-2.5-flash-image` (`GOOGLE_IMAGE_MODEL`) |
| API key | `GEMINI_API_KEY` |
| Approx. cost | `IMAGE_GENERATION_FEE_PER_CALL_USD` default **$0.039**/image |

**Analyze** uploaded images still uses OpenAI vision (`cost_usd_vision` on `ImageAnalyzerAgent`).

Only when `agent_mode=interactive` and query matches generate intent (e.g. “draw a photosynthesis diagram”). Response includes `generated_images[]` with **base64** (`data`) and a persistent **URL** (`url`), plus `image_id`. Files are stored under `GENERATED_IMAGE_DIR` (default `./data/generated`) and served at `{PUBLIC_BASE_URL}/generated/{session_id}/{image_id}.png`. Metadata is saved in Mongo per session. On reload, `GET /sessions/{user_session_id}/generated-images?user_id=...` returns URLs only (no base64). Routed via `image_generation_agent` (skips RAG/ReAct).

**Env:** `GENERATED_IMAGE_DIR`, `PUBLIC_BASE_URL` (e.g. `http://localhost:8001`), `MAX_REMEMBERED_GENERATED_IMAGES`. Docker: mount `./data/generated` (see `docker-compose.yml`).

### API cost fields (per `/chat` response)

- `cost_usd_llm`, `cost_usd_web_search`, `cost_usd_vision`, `cost_usd_image_generation`, `cost_usd_background`, `cost_usd_embedding`, `cost_usd_total`
- `web_search_calls`, `images_analyzed`, `image_generation_calls`, `generated_images`, `image_generation_backend`, `image_model_used`, `cost_by_component_usd`

Run `tests/benchmark_cost.py` after deploy (Case 4 — image generation). Set `GEMINI_API_KEY` in `.env` before testing.

---

## 3. What Happens in a Single Query (One Turn)

When a user sends one message, the system runs through multiple LLM calls in sequence via LangGraph:

```
User Query
    │
    ▼
[1] Language Detector (fastText — FREE, no API cost)
    │
    ▼
[2] Query Classifier      ← LLM call #1  (input: query + history)
    │
    ▼
[3] RAG Retriever         ← Embedding call (input: query text only)
    │
    ▼
[4] Main Agent (ReAct)    ← LLM call #2–3 (input: query + context + history)
    │    (may loop 1–3x depending on tool use)
    │
    ▼
[5] Groundedness Validator ← LLM call (if validation_mode = "strict"/"fast")
    │
    ▼
[6] Translator             ← LLM call (ONLY if language ≠ "en")
    │
    ▼
[7] Memory Saver           ← No LLM call (Redis write)
    │
    ▼
Response returned to user
```

**Background (async, after response):**
```
[8] Memory Summariser      ← LLM call (fires every ~5 turns to compress history)
```

---

## 4. Token Cost Formula

### Per Query Cost

```
Cost = (input_tokens × $0.15 / 1,000,000)
     + (output_tokens × $0.60 / 1,000,000)
     + (bg_input_tokens × $0.15 / 1,000,000)
     + (bg_output_tokens × $0.60 / 1,000,000)
     + (embedding_tokens × $0.13 / 1,000,000)
```

### Where Each Token Count Comes From

| Field | Source | What It Measures |
|:---|:---|:---|
| `input_tokens` | API response body | All tokens sent TO the model in this turn (prompt + history + RAG context) |
| `output_tokens` | API response body | All tokens generated BY the model in this turn |
| `bg_input_tokens` | Redis cache (`bg_tokens:{session_id}`) | Input tokens used by the background memory summariser |
| `bg_output_tokens` | Redis cache | Output tokens from the background summariser |
| `embedding_tokens` | Estimated ~20 tokens | Tokens in the query sent to Pinecone for similarity search |

---

## 5. Why Input Tokens Are the Biggest Cost Driver

Input tokens grow with every turn because the full conversation history is included in each prompt:

```
Turn 1:  System prompt (800t) + Query (15t) + RAG context (400t) = ~1,275 tokens
Turn 5:  System prompt (800t) + History (600t) + Query (15t) + RAG (400t) = ~1,781 tokens
Turn 10: System prompt (800t) + History (1,700t) + Query (15t) + RAG (400t) = ~2,893 tokens
Turn 15: System prompt (800t) + History (2,200t) + Query (15t) + RAG (400t) = ~3,396 tokens
```

> **Key insight:** The system prompt alone is ~800 tokens and repeats on every single turn. RAG context (retrieved curriculum chunks) adds ~300–600 tokens. These are the two largest fixed costs per turn.

---

## 6. Benchmark Results (Real Measured Data)

> Run date: 2026-04-06 | 15-turn conversation per case | Multi-subject (Biology, Physics, Math, History, Chemistry)

### Case 1 — English Only

| Metric | Value |
|:---|--:|
| Session total input tokens | 47,019 |
| Session total output tokens | 3,325 |
| Background tokens (input + output) | 3,566 |
| Embedding tokens (estimated) | 300 |
| **TOTAL session cost** | **$0.00972** |
| Average tokens per query | 3,614 |
| **Average cost per query** | **$0.000648** |
| Average latency per turn | 5.9s |

### Case 2 — Hindi (Translation Active)

| Metric | Value |
|:---|--:|
| Session total input tokens | 90,734 |
| Session total output tokens | 11,874 |
| Background tokens (input + output) | 6,266 |
| Embedding tokens (estimated) | 300 |
| **TOTAL session cost** | **$0.02198** |
| Average tokens per query | 7,278 |
| **Average cost per query** | **$0.001465** |
| Average latency per turn | 20.6s |

---

## 7. Why Hindi Costs 2.26× More than English

Three compounding factors:

### Factor 1 — Token Inflation (Devanagari Script)
Hindi text in Devanagari script uses ~2–3× more tokens per word compared to English. The same sentence that is 20 tokens in English may be 45–60 tokens in Hindi.

| Language | Avg output tokens/turn |
|:---|--:|
| English | 222 |
| Hindi | 792 |

### Factor 2 — Extra LLM Call (Translation)
Every Hindi turn fires the `TranslateResponseNode`, which makes one additional GPT-4o-mini call:
- Input: English response text (~200–600 tokens)
- Output: Hindi translated text (~500–1,000 tokens)

This adds ~1 LLM call per turn.

| Language | Avg LLM calls/turn |
|:---|--:|
| English | 2.5 |
| Hindi | 4.6 |

### Factor 3 — Cumulative History is Larger
Since Hindi outputs are verbose (~3.5× English), the conversation history accumulated in Redis is also larger, meaning each subsequent turn's `input_tokens` includes a bigger history block.

---

## 8. Scale Projections

### Formula for Any Scale

```
Total Cost = Number of Queries × Average Cost per Query

English: Total Cost = N × $0.000648
Hindi:   Total Cost = N × $0.001465
```

### Projection Table

| Queries | English Cost | Hindi Cost | Mixed (70/30) |
|--:|--:|--:|--:|
| 10,000 | $6.48 | $14.65 | $8.93 |
| 100,000 | $64.80 | $146.50 | $89.30 |
| 500,000 | $324.00 | $732.50 | $446.50 |
| **1,800,000** | **$1,166.40** | **$2,637.00** | **$1,608.78** |
| 5,000,000 | $3,240.00 | $7,325.00 | $4,465.00 |

> **In INR (@ ₹84/USD):**  
> 1.8M English = **₹97,978** | 1.8M Hindi = **₹2,21,508** | 1.8M Mixed = **₹1,35,137**

### Daily → Monthly Breakdown (1.8M queries/month)

| | English | Hindi |
|:---|--:|--:|
| Queries per day | 60,000 | 60,000 |
| Equivalent sessions/day (÷15 turns) | 4,000 | 4,000 |
| **Daily cost** | **$38.88** | **$87.90** |
| **Monthly cost (30 days)** | **$1,166** | **$2,637** |

---

## 9. Background Summarisation Cost (Memory Service)

The memory service fires automatically every ~5 turns to compress conversation history into a summary. This is an **async background task** — the user doesn't wait for it, but it does consume tokens.

| | English | Hindi |
|:---|--:|--:|
| BG tokens fired at turns | 6, 11 | 6, 11 |
| English BG tokens (session) | 3,566 | — |
| Hindi BG tokens (session) | — | 6,266 |
| BG cost as % of total | ~7% | ~10% |

> BG summarisation fires roughly every 5–6 turns. For very long conversations (>20 turns), this will fire 3–4 times per session.

---

## 10. Embedding Cost (RAG Retrieval)

Each user query is embedded and used to search Pinecone for relevant curriculum chunks. This uses `text-embedding-3-large`.

| Parameter | Value |
|:---|:---|
| Model | `text-embedding-3-large` |
| Price | $0.13 per 1M tokens |
| Avg tokens per query | ~20 tokens |
| Cost per query | ~$0.0000026 |
| Cost per 1.8M queries | **~$4.68** |

> Embedding cost is negligible (~0.4% of total) compared to the main LLM cost.

---

## 11. Cost Reduction Opportunities

| Optimisation | Expected Saving | Effort |
|:---|:---|:---|
| **Reduce `max_chunk_tokens` (800→500)** | ~20–25% less input | Low |
| **Trim memory history earlier** | ~15–20% less input | Low |
| **Prompt caching** (repeated system prompt) | ~30–40% less input cost via cached rate | Medium |
| **Switch Translator to Gemini 1.5 Flash** | ~50% less cost on translation calls | Medium |
| **Switch Validator to Gemini 1.5 Flash** | ~50% less on validation calls | Medium |
| **Combined optimised stack** | **~40–50% total reduction** | Medium |

### Optimised Monthly Estimate (1.8M queries)

| Scenario | Current | Optimised |
|:---|--:|--:|
| English | $1,166 | ~$700 |
| Hindi | $2,637 | ~$1,350 |

---

## 12. How to Re-Run the Benchmark

```bash
# Ensure the server is running (or use deployed URL)
uv run python tests/benchmark_cost.py
```

Results are saved to:  
`tests/benchmark_results_YYYYMMDD_HHMMSS.json`

To change the target URL, edit `BASE_URL` at the top of `tests/benchmark_cost.py`.
