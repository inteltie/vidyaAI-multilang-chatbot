# VidyaAI-AGENT: Quick Reference Guide

A one-page guide for new maintainers to quickly understand the system.

---

## 🎯 What Does This System Do?

An educational chatbot that answers student and teacher questions about curriculum by:
1. Analyzing the question (what language? what type of question?)
2. Retrieving relevant educational documents
3. Generating a tailored response
4. Validating the response isn't hallucinated
5. Translating back to user's language
6. Storing conversation for context

---

## 🏗️ High-Level Flow

```
User Asks Question
    ↓
Load Previous Chat History (Redis + MongoDB)
    ↓
Analyze Query (Language → English, Classify Type, Extract Context)
    ↓
Route to Agent (Conversational OR Educational)
    ↓
Retrieve Documents from Pinecone (Dense + Sparse Hybrid Search)
    ↓
Agent Generates Response (with Tools: Retrieval + Web Search)
    ↓
Validate Response (Against Documents, Check for Hallucinations)
    ↓
Translate Back to User's Language
    ↓
Save to Chat History (Redis + MongoDB)
    ↓
Return Response to User
```

---

## 🔧 Key Configuration (What You Can Adjust)

Located in `.env` file:

```env
# Response Length Limits
MAX_TOKENS_BRIEF=800              # Short answers
MAX_TOKENS_DEFAULT=1500           # Standard answers
MAX_TOKENS_DETAILED=3000          # Deep explanations

# Memory Settings
MEMORY_BUFFER_SIZE=20             # How many message turns to remember
MEMORY_TOKEN_LIMIT=2000           # Max tokens in memory

# Search Settings
RETRIEVER_TOP_K=5                 # How many documents to retrieve
RETRIEVER_SCORE_THRESHOLD=0.45    # Min match quality for retrieval (0-1)
CITATION_SCORE_THRESHOLD=0.6      # Min score for final citations

# Agent Settings
MAX_ITERATIONS=5                  # Max tool use loops
WEB_SEARCH_ENABLED=true          # Allow web search fallback

# LLM Settings
MODEL_NAME=gpt-4o-mini            # Which GPT model to use
LLM_TEMPERATURE=0.0               # 0 = deterministic, 1 = creative
```

---

## 📊 Request Processing (What Happens When)

### 1️⃣ LoadMemoryNode
- **When**: First step, always
- **What**: Loads conversation history
- **From**: Redis (fast) or MongoDB (persistent)
- **Outputs**: Session object, message buffer, conversation summary

### 2️⃣ AnalyzeQueryNode ⭐ (Merged Step - Does Multiple Things)
- **When**: Right after loading memory
- **What**: 
  - Detects user's language (English? Spanish? Hindi?)
  - Translates to English if needed
  - Classifies question type (chat vs curriculum)
  - Extracts metadata (class level, subject, chapter, lecture)
  - **Proactively fetches documents** (optimization)
- **Outputs**: `query_type`, `translated_query`, `documents`, `session_metadata`

### 3️⃣ Routing Decision
- **If** query_type = "conversational" (hi, thanks, hello) → **ConversationalAgent**
- **If** query_type = "curriculum_specific" (any education question):
  - **If** user_type = "teacher" → **TeacherAgent**
  - **If** agent_mode = "interactive" → **InteractiveStudentAgent**
  - **Otherwise** → **StudentAgent**

### 4️⃣ Agent Nodes (Student/Teacher/Conversational)
- **What**: Generate the answer
- **Tools Available**:
  - RetrievalTool: Search curriculum documents
  - WebSearchTool: Search internet (if docs insufficient)
- **Max Iterations**: 5 (prevents infinite loops)
- **Output**: `response` text

### 5️⃣ GroundednessCheckNode (Validation) 🛡️
- **When**: After educational agents only
- **What**: Checks if response is:
  - Supported by retrieved documents? (groundedness)
  - Answering the right question? (intent alignment)
  - Ambiguous? (needs user clarification?)
- **Outcomes**:
  - ✅ `is_valid=True` → Continue
  - ❌ `is_valid=False` → Retry agent (1x only)
  - ❓ `needs_clarification=True` → Ask user which interpretation

### 6️⃣ TranslateResponseNode
- **When**: Before returning to user
- **What**: Translates response back to user's original language
- **If**: User asked in Spanish, response is in Spanish (even if generated in English)

### 7️⃣ SaveMemoryNode
- **When**: At the very end
- **What**: Saves message to both storages
  - Redis: Immediate (for next request)
  - MongoDB: Background async task
- **Also**: Updates summary every 20 messages

---

## 💾 How Caching & Memory Works

### Three-Layer Storage

```
┌─────────────────────────────────────────────────────┐
│ 1. REDIS (Hot Cache - 1 hour TTL)                   │
│    - Current conversation buffer (last 30 messages) │
│    - Web search results (24 hour TTL)               │
│    - API response cache                             │
├─────────────────────────────────────────────────────┤
│ 2. MONGODB (Cold Storage - Permanent)               │
│    - Full message history                           │
│    - Session summaries (updated every 20 messages)  │
│    - Metadata and analytics                         │
├─────────────────────────────────────────────────────┤
│ 3. PINECONE (Vector DB - Permanent)                 │
│    - Educational documents with embeddings          │
│    - Searchable by: subject, chapter, class, etc.   │
└─────────────────────────────────────────────────────┘
```

### Token Budget Management

The system **never exceeds token limits** through trimming:

```
Trimming Strategy:
1. Include summary (context w/o tokens)
2. Add trimmed message history (up to 2000 tokens)
3. Add system prompt
4. Add current query
5. Total must fit in model's context window

If too many tokens: Drop oldest messages first
                    Keep most recent conversation
```

### Summaries (Generated Every 20 Messages)

```
OLD Summary: "User asked about photosynthesis and chlorophyll"
Last 20 Messages: [user: glucose?, assistant: glucose is..., etc.]
New Summary: "User learned photosynthesis, chlorophyll, glucose synthesis"
Stored in: MongoDB ChatSession document
Used in: LLM system prompt (doesn't count tokens)
```

---

## 🔍 How Retrieval & Search Works

### Retrieval Tool (RAG)

**Used For**: Educational questions about curriculum

**How It Works**:
```
1. Embed query using OpenAI embeddings (dense vector)
2. Create keyword index using BM25 (sparse vector)
3. Hybrid search: 80% dense + 20% sparse (alpha=0.8)
4. Filter by: class_level, subject, chapter, lecture_id (STRICT: only from request body)
5. Return top 5 documents (if score > 0.45)
6. Citations filtered at score > 0.6 for final display
```

**Documents Look Like**:
```json
{
  "id": "doc_12345",
  "score": 0.87,
  "text": "Photosynthesis is the process...",
  "metadata": {
    "lecture_id": 42,
    "subject": "Biology",
    "chapter": "Plant Processes",
    "class_level": "Class 10",
    "teacher_name": "Dr. Smith"
  }
}
```

### Web Search Tool

**Used For**: When RAG doesn't have enough info OR for current events

**How It Works**:
```
1. Check Redis cache first (24-hour TTL)
2. If not cached: Call OpenAI native web search
3. Get concise summary of web results
4. Cache result for 24 hours
```

**IMPORTANT**: Web search results are for **internal context only**. They are NEVER cited or mentioned to students. Only curriculum materials (Lecture IDs) are cited.

### When Each Is Used

| Situation | Tool | Reason |
|-----------|------|--------|
| Student asks about photosynthesis | Retrieval | Curriculum content |
| Student asks about 2024 election | Web Search | Not in curriculum |
| Teacher asks about pedagogy | Web Search | Latest methods |
| No retrieval results found | Web Search | Fallback |
| Conversational (hi, thanks) | Neither | Direct LLM response |

---

## 🤖 Four Types of Agents

### 1. ConversationalAgent
- **For**: Greetings, small talk, thanks
- **Tools**: None (no retrieval)
- **Max Tokens**: 800
- **Behavior**: Friendly, quick responses
- **Examples**: "Hi!", "How are you?", "Thanks for the help!"

### 2. StudentAgent (Standard) - With Grade Personas
- **For**: Student with curriculum questions
- **Tools**: Retrieval + Web Search (optional)
- **Max Tokens**: 1500
- **Behavior**: Adapts to student grade level (A/B/C/D)
- **Grade Personas**:
  - **Grade A**: "The Analytic Architect" - Technical depth, uses term "Kinetic Impedance", ends with "What if..." questions
  - **Grade B**: "The Structured Scholar" (Default) - Clear definitions, standard academic structure
  - **Grade C**: "The Helpful Neighbor" - Analogies like "sandpaper", real-world examples, encouraging
  - **Grade D**: "The Foundational Coach" - Simple stories, no jargon, "You've got this!" framing

### 3. InteractiveStudentAgent (Socratic)
- **For**: Student who wants to learn through questioning
- **Tools**: Retrieval + Web Search (optional)
- **Max Tokens**: 1500
- **Behavior**: Asks guiding questions instead of answers, also uses grade personas
- **Style**: "What do you already know about this? → Let's think about X → What does that tell you?"

### 4. TeacherAgent
- **For**: Teachers requesting pedagogical guidance
- **Tools**: Retrieval + Web Search (optional)
- **Max Tokens**: 3000
- **Behavior**: Scholarly, analytical, content review
- **Style**: "Coverage Analysis: You covered X in session Y. Topics: 1. A, 2. B, 3. C [Citations: session_10, session_12]"

---

## 🚨 Query Type Classification

### How It Works

1. **Check Heuristics** (fast, no LLM):
   - "hi", "hello", "thanks" → conversational

2. **LLM Classification** (if not obvious):
   - Analyzes: current query + last 4 messages
   - Decides: conversational OR curriculum_specific
   - Also extracts: class_level, subject, chapter, lecture_id

### Classification Result

```python
{
  "query_type": "curriculum_specific",           # or "conversational"
  "translated_query": "Explain photosynthesis",  # in English
  "confidence": 0.98,
  "subjects": ["Biology", "Science"],
  "class_level": "Class 10",
  "chapter": "Plant Processes",
  "lecture_id": "42"
}
```

---

## 📈 Performance Metrics to Track

### Key Numbers in `config.py`

| Metric | Default | Impact |
|--------|---------|--------|
| `retriever_top_k` | 5 | More = slower but more info |
| `retriever_score_threshold` | 0.45 | Higher = fewer but better docs |
| `citation_score_threshold` | 0.6 | Only high-quality citations shown |
| `max_iterations` | 5 | Max tool use loops |
| `memory_token_limit` | 2000 | More = better context, slower |
| `memory_buffer_size` | 20 | More = better context, slower |
| `web_search_enabled` | true | Enable/disable web fallback |

### Response Time Factors

1. **Embedding** (~500ms): Convert query to vector
2. **Retrieval** (~1s): Search Pinecone
3. **LLM Call** (~1-3s): Generate response
4. **Validation** (~1s): Check groundedness
5. **Translation** (~500ms): If needed

**Total**: ~3-6 seconds typical

---

## 🐛 Debugging Checklist

When something goes wrong:

```
□ Check .env variables are set correctly
□ Check Redis is running: redis-cli ping
□ Check MongoDB is running: mongo --eval "db.adminCommand('ping')"
□ Check OpenAI API key is valid
□ Check Pinecone index name and API key
□ Look at logs: docker-compose logs -f api
□ Check Redis cache: redis-cli KEYS "chat:*"
□ Check MongoDB docs: db.chatsessions.find({})
□ Verify document retrieval: Test retriever.py directly
```

---

## 📚 File Structure Cheat Sheet

```
main.py                      ← FastAPI app entry point
config.py                    ← Configuration & environment
graph.py                     ← LangGraph workflow (node connections)
state.py                     ← Data structures (AgentState, etc.)

services/
  ├─ query_classifier.py     ← Classify query type
  ├─ chat_memory.py          ← Redis + MongoDB memory management
  ├─ retriever.py            ← Pinecone hybrid search
  ├─ response_validator.py   ← Groundedness checking
  ├─ translator.py           ← Language translation
  └─ citation_service.py     ← Extract citations from documents

agents/
  ├─ student_agent.py        ← Standard student responses
  ├─ interactive_student_agent.py  ← Socratic questions
  ├─ teacher_agent.py        ← Teacher guidance
  ├─ conversational_agent.py  ← Chat responses
  └─ react_agent.py          ← ReAct loop (tool use)

nodes/
  ├─ load_memory.py          ← Load chat history
  ├─ analyze_query.py        ← Translate + classify + extract
  ├─ groundedness_check.py    ← Validate response
  ├─ translate_response.py    ← Translate to user language
  └─ save_memory.py          ← Save to Redis + MongoDB

tools/
  ├─ retrieval_tool.py       ← Search curriculum documents
  └─ web_search_tool.py      ← Search internet

models/
  ├─ chat.py                 ← ChatRequest, ChatResponse
  └─ domain.py               ← QueryIntent, ChatSession
```

---

## 🔗 Dependencies You Should Know

| Package | Role | Why |
|---------|------|-----|
| `langgraph` | Workflow orchestration | Manage node flow |
| `langchain` | LLM abstractions | Simplify OpenAI calls |
| `fastapi` | Web framework | HTTP API |
| `redis` | Caching | Fast memory |
| `motor` + `beanie` | MongoDB async | Persistent storage |
| `pinecone` | Vector search | Document retrieval |
| `openai` | LLM API | Generate responses |

---

## 💡 Key Insights for Maintenance

1. **Everything is async** - Never use `.get()` without `await`
2. **Tokens are expensive** - Always trim and summarize
3. **Cache aggressively** - Web results, embeddings, classifications
4. **Validate always** - Groundedness check prevents hallucinations
5. **Handle errors gracefully** - Fallback to web search if retrieval fails
6. **Monitor memory** - Watch Redis/MongoDB growth
7. **Temperature is 0** - Responses are deterministic (reproducible)
8. **Filters are strict** - User's class/subject filters are enforced

---

## 🚀 Common Tasks

### Add a New Agent Type
1. Create `agents/my_agent.py` (implement `Agent` protocol)
2. Create node in `nodes/my_agent_node.py`
3. Register in `graph.py` (add_node + add_edge)
4. Add routing logic in `_route_to_agent()` or `_route_educational_user()`

### Change Token Limits
1. Edit `.env`:
   ```
   MAX_TOKENS_DEFAULT=2000
   ```
2. Restart API

### Improve Retrieval Quality
1. Adjust in `.env`:
   ```
   RETRIEVER_TOP_K=10           # Get more docs
   RETRIEVER_SCORE_THRESHOLD=0.6  # Stricter matching
   ```
2. Test with sample queries

### Debug Memory Issues
1. Check Redis:
   ```bash
   redis-cli LRANGE chat:{session_id}:buffer 0 -1
   ```
2. Check MongoDB:
   ```bash
   db.chatsessions.findOne({session_id: "{session_id}"})
   ```

---

## 🎓 Learning Path for New Maintainers

1. **Day 1**: Read this guide + PROJECT_OVERVIEW.md
2. **Day 2**: Run locally, trace through a sample request
3. **Day 3**: Understand config.py + state.py
4. **Day 4**: Trace graph.py (node connections)
5. **Day 5**: Study memory management (chat_memory.py)
6. **Week 2**: Study agents and tools
7. **Week 3**: Study validation and error handling

---

**Last Updated**: January 2026
**Maintained By**: Your Team
**Questions?** Check inline code comments and PROJECT_OVERVIEW.md
