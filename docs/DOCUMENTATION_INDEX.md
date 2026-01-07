# Documentation Index

Complete documentation set for the VidyaAI-AGENT project. Start here.

---

## 📚 Documentation Files Overview

### 1. **QUICK_REFERENCE.md** ⭐ START HERE
- **Purpose**: One-page guide for new maintainers
- **Length**: ~500 lines
- **Best For**: Getting up to speed quickly
- **Covers**:
  - What the system does
  - High-level flow (4 minute read)
  - Configuration values you can adjust
  - Step-by-step request processing
  - How memory and caching work
  - The four agent types
  - Query classification rules
  - Debugging checklist
  - File structure reference
  - Common tasks and learning path

### 2. **PROJECT_OVERVIEW.md** ⭐ COMPREHENSIVE GUIDE
- **Purpose**: Complete system reference for maintainers
- **Length**: ~940 lines
- **Best For**: Understanding the entire architecture
- **Covers**:
  - Complete architecture with diagrams
  - Configuration & settings explained
  - Full request processing flow
  - Query type classification details
  - Three-layer caching system
  - Memory management with examples
  - Retrieval & search tools (how both work)
  - Response validation & token limits
  - Four agent types in detail
  - Message management & context
  - Key configuration parameters
  - Deployment & running instructions
  - Operational insights for maintainers

### 3. **TECHNICAL_DEEP_DIVE.md** 🔧 ADVANCED REFERENCE
- **Purpose**: Deep technical details for developers
- **Length**: ~1500 lines
- **Best For**: Understanding internals and troubleshooting
- **Covers**:
  - Full system architecture diagrams
  - Request processing pipeline (detailed flow)
  - Data flow through all services
  - Node-by-node breakdown (7 nodes explained)
  - Routing logic (2 conditional routers)
  - Agent implementation details
  - Query classification deep dive
  - Hybrid search internals (dense + sparse)
  - Token trimming algorithm
  - Summary generation process
  - ReAct loop with tool execution
  - Response validation logic
  - Distributed system considerations
  - Error handling & fallback chains
  - Performance optimization techniques
  - Deployment architecture
  - Monitoring & observability

### 4. **README.md** (Existing)
- **Purpose**: Quick start guide
- **Best For**: Initial setup and running locally

---

## 🎯 Which Document to Read When?

### New Developer (Day 1-2)
1. Read `QUICK_REFERENCE.md` completely (30 min)
2. Skim `PROJECT_OVERVIEW.md` sections 1-5 (30 min)
3. Run the system locally
4. Trace a sample request through the code

### Implementing a Feature
1. Check `QUICK_REFERENCE.md` "Common Tasks" section
2. Review `PROJECT_OVERVIEW.md` relevant sections
3. Read specific node/service code
4. Refer to `TECHNICAL_DEEP_DIVE.md` if needed for edge cases

### Debugging an Issue
1. Use `QUICK_REFERENCE.md` "Debugging Checklist"
2. Refer to `TECHNICAL_DEEP_DIVE.md` "Error Handling & Fallbacks"
3. Check file-specific logging and error patterns

### Performance Optimization
1. Read `TECHNICAL_DEEP_DIVE.md` section 10 (Optimization)
2. Review caching strategy in `PROJECT_OVERVIEW.md` section 5
3. Monitor metrics from section 12 (Monitoring)

### Deployment & DevOps
1. Start with `QUICK_REFERENCE.md` deployment section
2. Review `TECHNICAL_DEEP_DIVE.md` section 11 (Deployment)
3. Check Docker setup in docker-compose.yml

---

## 📖 Key Concepts Quick Reference

### Architecture Layers (top to bottom)
```
HTTP API (FastAPI) 
    ↓
LangGraph (Workflow)
    ↓
Services (Business Logic)
    ↓
Tools (Retrieval, Web Search)
    ↓
External APIs (OpenAI, Pinecone)
    ↓
Storage (Redis, MongoDB)
```

### Request Processing (8 nodes)
```
1. LoadMemory      - Fetch chat history
2. AnalyzeQuery    - Translate, classify, extract context, fetch docs
3. Route           - Pick conversational OR educational agent
4. Agent           - Generate response (with/without tools)
5. Validate        - Check groundedness & intent alignment
6. Translate       - Convert back to user's language
7. SaveMemory      - Store in Redis + MongoDB
```

### Four Agent Types
| Agent | User | Mode | Tool Use | Approach |
|-------|------|------|----------|----------|
| Conversational | Any | - | No | Friendly responses |
| Student | Student | Standard | Yes | Clear facts + examples |
| Interactive | Student | Interactive | Yes | Socratic questions |
| Teacher | Teacher | - | Yes | Scholarly analysis |

### Memory System
```
Redis (Hot)      → Last 30 messages, 1-hour TTL
MongoDB (Cold)   → All messages, permanent
Summary          → Every 20 messages, separate storage
Token Trimming   → Max 2000 tokens in conversation
```

### Caching Strategy
```
Web Search Results → 24-hour TTL
Translations       → 24-hour TTL
Session Buffer     → 1-hour TTL
Embeddings         → Permanent (Pinecone)
```

### Query Classification
```
Conversational  → Greetings, thanks, acknowledgments (no RAG)
Curriculum      → Educational questions (with RAG)
```

### Retrieval (Hybrid Search)
```
Dense Vector   (80%)  ← OpenAI text-embedding-3-large
Sparse Vector  (20%)  ← Pre-trained BM25 encoder
Combined Score        ← Alpha blend of both
```

---

## 🔧 Configuration & Adjustment Guide

All settings are in `.env` file:

### Token Limits (Adjust Response Length)
```env
MAX_TOKENS_BRIEF=800              # ← Increase for longer answers
MAX_TOKENS_DEFAULT=1500
MAX_TOKENS_DETAILED=3000
```

### Memory Settings (Adjust Context)
```env
MEMORY_BUFFER_SIZE=20             # ← How many turns to remember
MEMORY_TOKEN_LIMIT=2000           # ← Max tokens in context
```

### Retrieval Settings (Adjust Search Quality)
```env
RETRIEVER_TOP_K=5                 # ← More docs = more info
RETRIEVER_SCORE_THRESHOLD=0.4     # ← Higher = stricter matching
```

### Agent Settings (Adjust Behavior)
```env
MAX_ITERATIONS=5                  # ← Max tool use loops
WEB_SEARCH_ENABLED=true           # ← Allow web search fallback
```

### Model Settings (Change LLM)
```env
MODEL_NAME=gpt-4o-mini            # ← Which model to use
LLM_TEMPERATURE=0.0               # ← 0=deterministic, 1=creative
```

---

## 🚀 Quick Start Paths

### Run Locally (Development)
```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start services
redis-server
mongod

# 4. Run API
python -m uvicorn main:app --reload

# 5. Test
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "session_id": "session_1",
    "query": "What is photosynthesis?",
    "user_type": "student"
  }'
```

### Run with Docker (Production)
```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker-compose up --build -d

# 3. Verify
docker-compose logs -f api

# 4. Test
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{...}'

# 5. Stop
docker-compose down
```

---

## 📊 File Organization

```
VidyaAI_AI_v2/
├── 📄 Documentation (NEW - you are here)
│   ├── QUICK_REFERENCE.md          ← Start here
│   ├── PROJECT_OVERVIEW.md         ← Comprehensive
│   ├── TECHNICAL_DEEP_DIVE.md      ← Deep details
│   └── README.md                   ← Existing setup guide
│
├── 🚀 Application
│   ├── main.py                     ← FastAPI entry point
│   ├── config.py                   ← Configuration
│   ├── graph.py                    ← LangGraph workflow
│   ├── state.py                    ← Data structures
│   └── docker-compose.yml          ← Services definition
│
├── 📦 Services (Business Logic)
│   └── services/
│       ├── query_classifier.py     ← Query classification (LLM)
│       ├── chat_memory.py          ← Redis + MongoDB memory
│       ├── retriever.py            ← Pinecone hybrid search
│       ├── response_validator.py   ← Validation (LLM)
│       ├── translator.py           ← Language translation
│       ├── citation_service.py     ← Citation extraction
│       ├── cache_service.py        ← Redis caching
│       └── utils.py                ← Helper functions
│
├── 🤖 Agents (LLM Reasoning)
│   └── agents/
│       ├── student_agent.py        ← Standard student
│       ├── interactive_student_agent.py  ← Socratic
│       ├── teacher_agent.py        ← Scholarly
│       ├── conversational_agent.py ← Chat
│       ├── react_agent.py          ← ReAct loop (tools)
│       └── base.py                 ← Agent protocol
│
├── 🔗 Nodes (Graph Nodes)
│   └── nodes/
│       ├── load_memory.py          ← Load chat history
│       ├── analyze_query.py        ← Translate & classify
│       ├── groundedness_check.py   ← Validation
│       ├── translate_response.py   ← Language translation
│       ├── save_memory.py          ← Store conversation
│       └── *_agent_node.py         ← Agent wrapper nodes
│
├── 🛠️ Tools (Agent Tools)
│   └── tools/
│       ├── retrieval_tool.py       ← Search curriculum
│       ├── web_search_tool.py      ← Search internet
│       └── base.py                 ← Tool protocol
│
├── 📊 Models (Data Structures)
│   └── models/
│       ├── chat.py                 ← ChatRequest, ChatResponse
│       └── domain.py               ← QueryIntent, ChatSession
│
├── 🧪 Scripts (Development)
│   └── scripts/
│       ├── verify_agents.py        ← Test agents
│       ├── test_interactive_persona.py  ← Test interactive
│       └── ... (other verification scripts)
│
└── 📦 Configuration
    ├── .env                        ← Environment variables
    ├── pyproject.toml              ← Dependencies
    ├── bm25_encoder.json           ← Pre-trained encoder
    └── Dockerfile                  ← Container definition
```

---

## 🔍 Finding Things in the Code

### "Where is X?"

| What | Where | File |
|-----|-------|------|
| Query classification logic | Service | `services/query_classifier.py` |
| Memory management | Service | `services/chat_memory.py` |
| Retrieval search | Service | `services/retriever.py` |
| Response validation | Service | `services/response_validator.py` |
| LangGraph setup | Main | `graph.py` |
| State definition | Main | `state.py` |
| Configuration | Main | `config.py` |
| HTTP routes | Main | `main.py` |
| Student agent | Agent | `agents/student_agent.py` |
| Retrieval tool | Tool | `tools/retrieval_tool.py` |
| Web search tool | Tool | `tools/web_search_tool.py` |

### "How does X work?"

| What | Documentation | File |
|-----|--------------|------|
| Hybrid retrieval | TECHNICAL_DEEP_DIVE section 4 | `services/retriever.py` |
| Query classification | PROJECT_OVERVIEW section 4 | `services/query_classifier.py` |
| Memory trimming | TECHNICAL_DEEP_DIVE section 5 | `services/chat_memory.py` |
| ReAct loop | TECHNICAL_DEEP_DIVE section 6 | `agents/react_agent.py` |
| Validation | TECHNICAL_DEEP_DIVE section 7 | `services/response_validator.py` |
| Routing | PROJECT_OVERVIEW section 9 | `graph.py` |

---

## 💡 Pro Tips for Maintainers

### Always Remember
- ✅ Everything is async (use `await`)
- ✅ Token efficiency matters (monitor usage)
- ✅ Cache aggressively (Redis is fast)
- ✅ Validate always (prevent hallucinations)
- ✅ Fail gracefully (fallbacks for errors)

### Common Mistakes to Avoid
- ❌ Using `.get()` without `await` on async functions
- ❌ Ignoring token limits (expensive & slow)
- ❌ Removing validation steps (causes hallucinations)
- ❌ Hard-coding filters (breaks flexibility)
- ❌ Synchronous operations in FastAPI (blocks requests)

### Testing Checklist
- [ ] Test with English query
- [ ] Test with non-English query
- [ ] Test conversational ("Hi!")
- [ ] Test educational ("Explain X")
- [ ] Test with invalid user_id
- [ ] Test with missing filters
- [ ] Verify Redis cache is working
- [ ] Verify MongoDB saves message
- [ ] Check token usage

### Performance Tuning
- Increase `RETRIEVER_TOP_K` if missing info
- Increase `MAX_TOKENS_*` if responses too brief
- Decrease `MEMORY_BUFFER_SIZE` if latency high
- Enable/disable `WEB_SEARCH_ENABLED` based on accuracy
- Monitor embedding latency (embedding generation)

---

## 📞 Support & Escalation

### If You Get Stuck

1. **Check QUICK_REFERENCE.md** "Debugging Checklist"
2. **Search PROJECT_OVERVIEW.md** for the section
3. **Review TECHNICAL_DEEP_DIVE.md** for internals
4. **Check inline code comments** (they're detailed)
5. **Review error logs** (use `docker-compose logs -f api`)

### Common Issues & Solutions

See QUICK_REFERENCE.md "Common Issues & Solutions" table for:
- "No documents found"
- "High latency"
- "Hallucinations"
- "Language not detected"
- "LLM rate limits"

---

## 📝 Version & Updates

- **Created**: January 2026
- **Last Updated**: January 2026
- **Documentation Version**: 1.0
- **System Version**: VidyaAI-AGENT v2

---

## 🎓 Learning Recommended Order

### Week 1: Foundation
- Day 1: QUICK_REFERENCE.md + run locally
- Day 2: PROJECT_OVERVIEW.md sections 1-5
- Day 3: PROJECT_OVERVIEW.md sections 6-12
- Day 4: Trace code for one request
- Day 5: Run debugger, set breakpoints

### Week 2: Deep Dive
- Day 6-7: TECHNICAL_DEEP_DIVE.md sections 1-3
- Day 8-9: TECHNICAL_DEEP_DIVE.md sections 4-7
- Day 10: TECHNICAL_DEEP_DIVE.md sections 8-12

### Week 3+: Contribution Ready
- Implement a small feature
- Write tests
- Deploy changes
- Monitor performance

---

**Next Steps**: 
1. Open QUICK_REFERENCE.md
2. Read through completely (30 min)
3. Run the system locally
4. Trace your first request
5. You're ready to maintain! 🚀

---

**Questions?** Review the relevant documentation file above. If still stuck, check the inline code comments - they're very detailed.
