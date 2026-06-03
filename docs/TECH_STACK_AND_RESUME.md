# VidyaAI — Tech Stack & Resume Reference

**Project:** VidyaAI (VidyaAI-AGENT) — Multilingual educational AI tutor with RAG, multi-agent orchestration, vision, and image generation.

**One-line summary (resume):** Built a production-grade AI tutoring platform using FastAPI, LangGraph, hybrid RAG (Pinecone + BM25), and multi-model LLM orchestration (OpenAI + Google Gemini) with Redis/MongoDB memory, Docker deployment, and CI/CD to AWS EC2.

---

## Core Technologies

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.12 |
| **Web framework** | FastAPI, Uvicorn (ASGI) |
| **Package manager** | uv (Astral) |
| **Validation / config** | Pydantic v2, python-dotenv |
| **Async runtime** | asyncio, httpx |
| **Testing** | pytest |

---

## AI / LLM Stack

| Component | Technology / Models |
|-----------|---------------------|
| **Agent orchestration** | LangGraph (`StateGraph`), LangChain |
| **Primary chat model** | OpenAI GPT-4o-mini |
| **Auxiliary models** | GPT-4.1-nano (query analysis, validation, translation, summarization) |
| **Embeddings** | OpenAI `text-embedding-3-large` (1536-dim) |
| **Agent pattern** | ReAct (Reasoning + Acting) with tool calling |
| **Image generation** | Google Gemini (`gemini-3.1-flash-image` via `google-genai` SDK) |
| **Vision (image input)** | OpenAI GPT-4o-mini multimodal (image analyzer agent) |
| **Web search** | OpenAI Responses API / web search tool |
| **Tokenization** | tiktoken |

### Multi-agent system

| Agent | Role |
|-------|------|
| **Student Agent** | Standard tutoring — fact-first answers with RAG + optional web search |
| **Interactive Student Agent** | Socratic, step-by-step learning; supports image upload analysis |
| **Teacher Agent** | Scholarly analysis for educator users |
| **Conversational Agent** | Greetings, small talk, off-topic handling |
| **Image Generation Agent** | Context-aware diagram/illustration generation (interactive mode) |
| **Image Analyzer Agent** | Vision pipeline for uploaded images |
| **Image Context Gate Agent** | Policy gate for when image context is injected |
| **Translation Agent** | Response translation with language policy |

---

## RAG & Search

| Feature | Implementation |
|---------|----------------|
| **Vector database** | Pinecone (hybrid index) |
| **Retrieval strategy** | Hybrid search — dense embeddings + BM25 sparse vectors (`pinecone-text`) |
| **Hybrid scaling** | Convex combination (alpha-weighted dense + sparse query vectors) |
| **Quality gating** | Score threshold, high/low RAG quality routing |
| **Web fallback** | Reactive web search when RAG confidence is low |
| **Query rewriting** | LLM-based web query rewriter with overlap guardrails |
| **Caching** | Redis-backed embedding cache; query classification cache |
| **Parallel retrieval** | Concurrent RAG + web search (`asyncio.gather`) |

---

## Data & Memory

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Hot cache** | Redis 7 | Conversation buffer, query/embedding cache, background token counters |
| **Persistent store** | MongoDB Atlas + Motor (async driver) | Sessions, messages, summaries, image metadata |
| **ODM** | Beanie | MongoDB document models (`ChatSession`, `ChatMessage`) |
| **Memory strategy** | Redis-first load with Mongo fallback; graceful degradation on Atlas timeout |
| **Summarization** | Background LLM summarization of long sessions |
| **Generated assets** | Local filesystem (`data/generated/`) + static file serving |

---

## API & Backend Features

### REST endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /chat` | Main conversational endpoint (student/teacher, standard/interactive modes) |
| `GET /health` | Health check |
| `GET /sessions/{id}/generated-images` | List generated diagram URLs for a session |
| `GET /sessions/{id}/messages` | Message history with image input/output modality flags |
| `GET /generated/{session}/{image_id}` | Static serving of generated images |

### Notable capabilities

- **Multilingual:** FastText language detection; automatic translation to/from user language
- **Query classification:** Intent routing (curriculum, conversational, image generation, off-topic)
- **Groundedness validation:** LLM-based hallucination / language consistency checks with retry
- **Citation system:** Lecture/transcript-aware citations from RAG metadata
- **Cost tracking:** Per-request token usage and USD cost breakdown by component (LLM, web, vision, image gen, embeddings)
- **Orchestration routing:** Explicit route decisions with confidence and fallback reasons
- **Image modality:** Image upload (vision), image generation (Gemini), interactive exploration zones
- **Grade-adaptive personas:** A/B/C/D teaching styles based on student proficiency
- **Session restart detection:** 2-hour inactivity warm re-greeting
- **CORS, structured errors, request timeouts:** Production-oriented HTTP handling

---

## Architecture Patterns

```
Client → FastAPI → LangGraph StateGraph → Nodes → Agents → Tools → External APIs
                      ↓
              Redis / MongoDB / Pinecone / OpenAI / Gemini
```

| Pattern | Where used |
|---------|------------|
| **State machine workflow** | LangGraph nodes: load_memory → analyze_query → route_decision → agent → groundedness → translate → save_memory |
| **Conditional routing** | Query type, user type (student/teacher), agent mode (standard/interactive), image modality |
| **Tool registry** | ReAct agents call retrieval and web search tools dynamically |
| **Planner + executor** | Image prompt composer plans diagram spec; Gemini tool executes generation |
| **Graceful degradation** | Mongo timeout → Redis-only memory; classification timeout → fallback routing |
| **Tiered model routing** | Primary vs nano models by task (chat vs classify vs validate) |
| **Factory / dependency injection** | `BackendApp` wires services, graph, and routes at startup |

---

## DevOps & Infrastructure

| Item | Details |
|------|---------|
| **Containerization** | Docker, Docker Compose |
| **Base image** | `python:3.12-slim` |
| **Process model** | Uvicorn with configurable workers |
| **CI/CD** | GitHub Actions → SSH deploy to AWS EC2 |
| **Health checks** | Docker healthcheck on `/health` |
| **Environment config** | `.env` + Pydantic settings (12-factor style) |
| **Logging** | Structured Python logging with timing breakdowns per graph node |

---

## Supporting Libraries & Tools

| Library | Use |
|---------|-----|
| **NLTK** | Text processing (transitive dependency / tokenizer warmup) |
| **FastText** | Offline language identification (`lid.176.bin`) |
| **NumPy** | Vector / numeric operations |
| **Beanie + Motor** | Async MongoDB ODM |
| **redis (asyncio)** | Async Redis client |

---

## Project Structure (high level)

```
VIDHYA_AI_v2/
├── main.py              # FastAPI app, lifespan, routes
├── graph.py             # LangGraph workflow builder
├── config.py            # Pydantic settings
├── state.py             # AgentState TypedDict
├── agents/              # ReAct & specialized agents
├── nodes/               # LangGraph node wrappers
├── services/            # RAG, memory, classification, costing, translation
├── tools/               # Retrieval, web search, image generation
├── models/              # Pydantic API models + Beanie documents
├── docs/                # Architecture & cost documentation
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/   # Deploy pipeline
```

---

## Resume Bullet Points (copy-ready)

Use 2–4 of these tailored to your role:

1. **Architected and implemented** a multi-agent educational AI platform using **LangGraph** and **FastAPI**, orchestrating 8+ specialized agents with conditional routing for students, teachers, and interactive learning modes.

2. **Built hybrid RAG pipeline** with **Pinecone** (dense + BM25 sparse vectors), **OpenAI embeddings**, quality-based web search fallback, and Redis caching — reducing latency and improving answer relevance for curriculum Q&A.

3. **Designed hybrid memory system** combining **Redis** hot-cache and **MongoDB Atlas** persistence with background summarization, session restart detection, and graceful degradation when the database is unreachable.

4. **Integrated multimodal AI**: OpenAI vision for uploaded image analysis and **Google Gemini** for context-aware educational illustration generation with planner-driven prompts and exploration-zone metadata for interactive UI.

5. **Implemented production observability**: per-request token usage, component-level USD cost tracking, graph node timing, routing diagnostics, and configurable timeouts for LLM, image generation, and full request lifecycle.

6. **Deployed containerized service** with **Docker Compose**, GitHub Actions CI/CD to **AWS EC2**, health checks, and environment-driven configuration via Pydantic.

7. **Added multilingual support** using FastText language detection and LLM-based translation with policy-driven skip/translate decisions for 170+ language scenarios.

8. **Engineered groundedness validation** layer with LLM response verification, retry logic, and citation recovery to reduce hallucinations in educational content.

---

## Skills Keywords (ATS / LinkedIn)

**Languages:** Python  

**Frameworks & APIs:** FastAPI, LangChain, LangGraph, Pydantic, Uvicorn  

**AI / ML:** RAG, ReAct agents, LLM orchestration, prompt engineering, hybrid search, embeddings, multimodal AI, vision LLM, image generation  

**Databases & cache:** Redis, MongoDB, Pinecone, Beanie, Motor  

**Cloud & DevOps:** Docker, Docker Compose, GitHub Actions, AWS EC2, CI/CD  

**Concepts:** Multi-agent systems, state machines, async I/O, microservices, REST APIs, session management, cost optimization, graceful degradation  

**External APIs:** OpenAI API, Google Gemini API, Pinecone API  

---

## Metrics & Scale Signals (for interviews)

- **Graph pipeline:** 10+ LangGraph nodes with parallel classification and retrieval paths
- **Model tiering:** 4+ distinct OpenAI models routed by task (chat, nano for classify/validate/translate/summary)
- **Timeout budgets:** Configurable graph (120s), image gen (90s), Mongo session load (3s), classification (15s)
- **Memory:** Redis buffer + Mongo persistence + structured session summaries
- **Modes:** Standard vs interactive tutoring; student vs teacher personas; text vs image input vs image generation

---

## Related Documentation

| Document | Contents |
|----------|----------|
| `README.md` | Quick start, features, architecture overview |
| `docs/PROJECT_OVERVIEW.md` | Detailed system design and configuration |
| `docs/TECHNICAL_DEEP_DIVE.md` | In-depth technical flows |
| `docs/COST_ANALYSIS.md` | Token and API cost breakdown |
| `DOCKER.md` | Container deployment guide |

---

*Last updated: May 2026 — reflects VidyaAI v2 codebase (FastAPI + LangGraph + RAG + Gemini image generation).*
