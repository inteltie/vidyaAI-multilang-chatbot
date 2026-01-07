# VidyaAI-AGENT: Comprehensive Project Overview

## Executive Summary

VidyaAI-AGENT is an intelligent educational chatbot powered by LangGraph that provides personalized learning experiences. It uses a multi-agent architecture with RAG (Retrieval-Augmented Generation) capabilities, advanced caching, and multilingual support. The system is built on FastAPI, uses OpenAI LLMs, Pinecone for vector search, Redis for caching/memory, and MongoDB for persistent storage.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Configuration & Settings](#configuration--settings)
3. [Request Flow & Processing](#request-flow--processing)
4. [Query Type Classification](#query-type-classification)
5. [Caching System](#caching-system)
6. [Memory Management](#memory-management)
7. [Retrieval & Search Tools](#retrieval--search-tools)
8. [Response Generation & Validation](#response-generation--validation)
9. [Agent Types & Routing](#agent-types--routing)
10. [Message & Context Management](#message--context-management)
11. [Key Configuration Parameters](#key-configuration-parameters)
12. [Deployment & Running](#deployment--running)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                      (main.py entry point)                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼──────┐          ┌─────▼────────┐
    │   Redis   │          │   MongoDB    │
    │  (Cache & │          │ (Persistent  │
    │  Sessions)│          │   Storage)   │
    └───────────┘          └──────────────┘
         │
    ┌────▼──────────────────────────────┐
    │    LangGraph Workflow (graph.py)   │
    │                                    │
    │  ┌──────────────────────────────┐ │
    │  │  1. LoadMemory Node          │ │
    │  │  2. AnalyzeQuery Node        │ │
    │  │  3. Agent Routing            │ │
    │  │     - Conversational Agent   │ │
    │  │     - Student Agent          │ │
    │  │     - Interactive Student    │ │
    │  │     - Teacher Agent          │ │
    │  │  4. Groundedness Check       │ │
    │  │  5. Translate Response       │ │
    │  │  6. Save Memory              │ │
    │  └──────────────────────────────┘ │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   External Services                │
    │   ├─ OpenAI GPT-4o-mini (LLM)     │
    │   ├─ Pinecone (Vector Search)     │
    │   └─ Web Search (via OpenAI)      │
    └────────────────────────────────────┘
```

### Core Components

| Component | Purpose | Key Details |
|-----------|---------|-------------|
| **FastAPI App** | HTTP request handling | Async, CORS enabled |
| **LangGraph** | Workflow orchestration | StateGraph with conditional routing |
| **Services** | Business logic layer | Query classification, memory, retrieval, validation |
| **Agents** | LLM-powered reasoning | ReAct with tool use (retrieval, web search) |
| **Tools** | External integrations | Retrieval Tool, Web Search Tool |
| **Nodes** | Graph nodes | 8 nodes that form the processing pipeline |

---

## Configuration & Settings

All configuration is managed through `config.py` using environment variables via **Pydantic**.

### Settings Class Structure

```python
class Settings(BaseModel):
    # External Services
    openai_api_key: str
    redis_url: str
    pinecone_api_key: str
    pinecone_index: str
    mongo_uri: str
    mongo_db_name: str
    
    # Model Settings
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    llm_temperature: float = 0.0
    
    # Application Settings
    max_tokens_*: int        # Token limits for responses
    memory_buffer_size: int  # Conversation history turns
    memory_token_limit: int  # Max tokens in memory
    max_iterations: int      # ReAct agent iterations
    web_search_enabled: bool # Enable/disable web search
    retriever_top_k: int     # Documents to retrieve
    retriever_score_threshold: float
```

### Loading Configuration

- Environment variables are loaded from `.env` file via `python-dotenv`
- A global `settings` instance is created at module load time
- Missing required fields raise `ValidationError`

---

## Request Flow & Processing

### High-Level Request Flow

```
User Request
    │
    ├─→ [1] LoadMemoryNode
    │       └─ Fetch conversation history from Redis/MongoDB
    │       └─ Return session object, buffer, summary
    │
    ├─→ [2] AnalyzeQueryNode (MERGED STEP)
    │       ├─ Language detection & English translation
    │       ├─ Query classification (conversational vs curriculum_specific)
    │       ├─ Intent classification
    │       ├─ Context extraction (class_level, subject, chapter, lecture_id)
    │       └─ PROACTIVE RAG FETCH (optimization: fetch docs during analysis)
    │
    ├─→ [3] Conditional Routing (route_to_agent)
    │       ├─ IF query_type == "conversational" → Conversational Agent
    │       └─ IF query_type == "curriculum_specific" → Route by user_type
    │
    ├─→ [4] Agent Processing (One of these):
    │       ├─ Conversational Agent (no tools, simple response)
    │       ├─ Student Agent (standard: fact-first synthesis)
    │       ├─ Interactive Student Agent (socratic: step-by-step)
    │       └─ Teacher Agent (scholarly analysis)
    │
    ├─→ [5] GroundednessCheckNode
    │       ├─ Validate response against retrieved documents
    │       ├─ Check intent alignment
    │       └─ Detect ambiguity (ask for clarification if needed)
    │
    ├─→ [6] TranslateResponseNode
    │       └─ Translate back to user's original language (if not English)
    │
    ├─→ [7] SaveMemoryNode
    │       ├─ Redis: Add to session buffer
    │       └─ MongoDB: Async background save + summary update
    │
    └─→ Return Response to Client
```

### Node Execution Order

The graph follows this flow:

1. **load_memory** → **analyze_query** (linear)
2. **analyze_query** → (conditional routing) → agent nodes
3. Agent nodes → **groundedness_check** (for educational) or skip
4. **groundedness_check** → **translate_response** (linear)
5. **translate_response** → **save_memory** → END

---

## Query Type Classification

### Classification Logic

The **AnalyzeQueryNode** uses `QueryClassifier` to determine query type.

#### Query Types

| Type | Description | Agent Path | Example |
|------|-------------|-----------|---------|
| **conversational** | Greetings, small talk, general help | Conversational Agent | "Hi!", "Thanks!", "Help me study" |
| **curriculum_specific** | Any educational question | Educational Agent (student/teacher) | "Explain photosynthesis", "What is DNA?" |

#### Classification Process

1. **Heuristics Check** (optional fast path)
   - Keywords: "hi", "hello", "thanks", "help", "goodbye"
   - Quick pattern matching without LLM

2. **LLM Classification** (full analysis)
   - Uses GPT-4o-mini with structured output
   - Inputs: current query + conversation history (last 4 turns)
   - Outputs: `QueryClassification` object

3. **Context Extraction** (simultaneous with classification)
   - Scans query and history for metadata:
     - `class_level`: "Class 10", "Grade 12", "Batch A"
     - `extracted_subject`: "Algebra", "Organic Chemistry"
     - `chapter`: "Quadratic Equations", "Chapter 5"
     - `lecture_id`: "session_12", "76"
   - Falls back to "General" if no subjects detected

### QueryClassification Output

```python
class QueryClassification(BaseModel):
    query_type: Literal["conversational", "curriculum_specific"]
    translated_query: str  # English translation
    confidence: float      # 0.0 - 1.0
    subjects: List[str]    # ["Math", "Science", etc.]
    class_level: Optional[str]
    extracted_subject: Optional[str]
    chapter: Optional[str]
    lecture_id: Optional[str]
```

---

## Caching System

The system implements **multi-layered caching** for performance optimization.

### 1. Redis Cache Service (`CacheService`)

#### Purpose
- Short-term caching for embeddings, web search results, and API responses
- Session buffer storage (conversation history)

#### Key Operations

```python
# Cache Operations
cache_key = CacheService.generate_key("web_search", query)
cached_result = await CacheService.get(cache_key)
await CacheService.set(cache_key, result, ttl=86400)  # 24-hour TTL

# Session Buffer
redis_key = f"chat:{session_id}:buffer"
await redis.rpush(redis_key, json.dumps(msg))
await redis.ltrim(redis_key, -30, -1)  # Keep last 30 turns
await redis.expire(redis_key, 3600)    # 1-hour session TTL
```

#### Cache Expiration Strategy

| Resource | TTL | Reason |
|----------|-----|--------|
| Web Search Results | 24 hours | Facts change slowly |
| Session Buffer | 1 hour | Real-time conversation |
| API Responses | 1-6 hours | Varies by query type |
| Embeddings | Permanent | Static content |

### 2. Conversation Memory Buffer (Redis + MongoDB)

#### How It Works

**Redis (Hot Storage)**
- Stores active conversation history
- Format: List of JSON message objects
- Last N turns (30 max) for quick access

**MongoDB (Cold Storage)**
- Persistent storage of all messages
- Full message history per session
- Indexed for fast lookups

#### Buffer Management

```python
# Ensure Session exists & load buffer
session, buffer, summary = await memory_service.ensure_session(user_id, session_id)

# Get context (trimmed by tokens)
summary, trimmed_history = await memory_service.get_context(session_id)

# Add message to both storages
await memory_service.add_message(session_id, role="user", content=query)
asyncio.create_task(memory_service.background_save_message(...))
```

### 3. Summary Generation & Caching

#### When Summaries Are Generated

- **Trigger**: Every 20 new messages in a session
- **Method**: Incremental summarization (previous summary + last 20 messages)
- **Storage**: MongoDB session document

#### Summary Usage in LLM Prompt

Summaries are included in the LLM system prompt as **context without consuming token budget**:

```
System Prompt:
"You are an educational tutor. 
Here's a brief summary of what we've discussed so far: {summary}
Current conversation history: {trimmed_history}
...respond to: {query}"
```

This provides **long-term context** while keeping **token consumption bounded**.

---

## Memory Management

### Memory Architecture

```
Session Lifecycle:
    1. User starts conversation → create ChatSession (MongoDB)
    2. Load summary from MongoDB
    3. Load recent history from Redis buffer
    4. During chat: maintain in Redis (fast)
    5. Every message: async save to MongoDB
    6. Every 20 messages: update summary
    7. Session expires: Redis buffer deleted after 1 hour TTL
```

### Key Concepts

#### ConversationTurn
```python
class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]  # Who sent the message
    content: str                         # Message text
```

#### Memory Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_buffer_size` | 20 | Number of turns to keep |
| `memory_token_limit` | 2000 | Max tokens in trimmed history |
| `Redis TTL` | 3600 | Session buffer expiration (1 hour) |

#### Token-Based Trimming

```python
# From MemoryService.get_context()
trimmed_history = trim_messages(
    messages,
    max_tokens=2000,           # Limit to 2000 tokens
    strategy="last",           # Keep most recent messages
    token_counter=llm,         # Use model's tokenizer
    start_on="human",          # Start with user message
    include_system=False       # System prompt separate
)
```

**Result**: Historical context is preserved up to token limit, oldest messages dropped first.

### Background Save Process

```python
async def background_save_message(session_id, user_id, role, content):
    # Add message to MongoDB
    session = await ChatSession.find_one(...)
    await session.add_message(role, content)
    
    # Update summary every 20 messages
    if len(session.messages) % 20 == 0:
        asyncio.create_task(background_update_summary(session_id))
```

---

## Retrieval & Search Tools

### Two-Tool Architecture

The system uses **two complementary tools** for information gathering:

#### 1. Retrieval Tool (RAG)

**Purpose**: Fetch curriculum-based documents from vector database

**Trigger**: 
- Educational queries (curriculum_specific)
- Retrieved proactively during AnalyzeQuery node
- Can be called again in ReAct agent loop

**Process**:

```python
async def execute(query: str, filters: Dict = None):
    # 1. Generate dense embedding
    dense = await self._embed(query_en)
    
    # 2. Generate sparse vector (BM25)
    sparse = self._bm25_encoder.encode_queries(query_en)
    
    # 3. Scale vectors (hybrid search)
    scaled_dense, scaled_sparse = _hybrid_scale(dense, sparse, alpha=0.8)
    
    # 4. Search Pinecone with user filters
    results = await self._index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        filter=filters,  # class_level, subject, chapter, lecture_id
        top_k=5,        # retriever_top_k
        include_metadata=True
    )
    
    # 5. Format and return documents
    return [Document(id, score, text, metadata)]
```

**Hybrid Search Details**:
- **Dense (80%)**: OpenAI text-embedding-3-large (1536 dims)
- **Sparse (20%)**: BM25 keyword-based vectors
- **Alpha**: 0.8 (weights dense more heavily)

**Filters Applied**:
- `class_level`: Filter by educational level
- `subject`: Filter by subject area
- `chapter`: Filter by chapter/topic
- `lecture_id`: Filter by specific lecture/session
- All filters come from user request (strict enforcement)

**Output Documents**:
```python
class Document(TypedDict):
    id: str              # Unique document ID
    score: float         # Similarity score (0.4 threshold)
    text: str            # Document content
    metadata: Dict       # lecture_id, subject, chapter, etc.
```

#### 2. Web Search Tool

**Purpose**: Search current web information for topics NOT in curriculum

**Trigger**:
- When RAG returns insufficient results
- For real-time facts (current events, latest discoveries)
- User-requested or agent decides

**Process**:

```python
async def execute(query: str):
    # Check Redis cache first
    cache_key = CacheService.generate_key("web_search", query)
    cached = await CacheService.get(cache_key)
    if cached: return cached
    
    # Call OpenAI chat completions with system prompt
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            system="Provide concise summary with sources",
            user=f"Briefly summarize: {query}"
        ],
        max_tokens=300,
        temperature=0.0
    )
    
    # Cache for 24 hours (TTL for web facts)
    result = f"WEB_SEARCH_OBSERVATION:\n{response}"
    await CacheService.set(cache_key, result, ttl=86400)
    return result
```

**Cache Strategy**: Web results cached for 24 hours (facts relatively stable)

### Tool Selection Decision Logic

| Condition | Tool Used | Reason |
|-----------|-----------|--------|
| Educational query + curriculum docs available | Retrieval Tool | Accurate curriculum-based answers |
| Educational query + no curriculum docs | Web Search Tool | Fallback for uncovered topics |
| Current events / real-time facts | Web Search Tool | Static curriculum can't cover current info |
| Small talk / conversational | Neither | LLM responds directly |

---

## Response Generation & Validation

### Token Limits by Response Type

The system adjusts response length based on context:

```python
# From config.py
max_tokens_brief = 800        # Quick factual answers
max_tokens_default = 1500     # Standard educational responses
max_tokens_detailed = 3000    # Deep dives with examples
```

#### How Response Type Is Determined

```python
# In Agent nodes
detail_level = extract_from_request(state)  # Brief / Default / Detailed
max_tokens = {
    "brief": 800,
    "default": 1500,
    "detailed": 3000
}.get(detail_level, 1500)

# Pass to LLM
response = await llm.ainvoke(
    prompt=system_prompt + user_query,
    max_tokens=max_tokens,
    temperature=0.0  # Deterministic for educational content
)
```

### Groundedness Validation Process

After agent generates response, **GroundednessCheckNode** validates it:

```python
class ValidationResult(BaseModel):
    is_valid: bool                    # Response supported by docs?
    needs_clarification: bool         # Ambiguous? Ask user?
    reasoning: str                    # Why is it valid/invalid?
    feedback: Optional[str]           # Correction if invalid
    clarification_question: Optional[str]  # Question if ambiguous
```

#### Validation Flow

```
1. Agent generates response
    ↓
2. Validator checks:
   - Is response grounded in retrieved documents?
   - Does it match the detected intent subject?
   - Are there multiple valid interpretations?
    ↓
3. Results:
   - is_valid=True → Proceed to translation
   - is_valid=False → Provide feedback, retry with correction
   - needs_clarification=True → Ask user which interpretation
    ↓
4. Max 1 correction retry (prevent infinite loops)
```

#### Ambiguity Detection Example

```
Query: "Explain transformers"
Documents Found:
  - Electrical transformers (power systems)
  - Transformer neural networks (AI/ML)

Agent Response: "A transformer is a neural network architecture..."

Validator Decision:
  needs_clarification = True
  clarification_question = "I found info on both electrical and neural network 
                           transformers. Which would you like explained?"
```

---

## Agent Types & Routing

### Agent Classification

```
                         User Query
                              │
              ┌───────────────┴────────────────┐
              │                                │
         [ConversationalAgent]        [Educational Agent]
         (Greetings, thanks)          (Curriculum questions)
              │                                │
              │                    ┌───────────┼───────────┐
              │                    │           │           │
                              [Student]   [Interactive]   [Teacher]
                              (Fact-first) (Socratic)   (Scholarly)
```

### Routing Decision Tree

```python
@staticmethod
def _route_to_agent(state):
    query_type = state.get("query_type")
    if query_type == "conversational":
        return "conversational"
    else:
        return "educational"

@staticmethod
def _route_educational_user(state):
    user_type = state.get("user_type")     # "student" or "teacher"
    agent_mode = state.get("agent_mode")   # "standard" or "interactive"
    
    if user_type == "teacher":
        return "teacher"
    elif agent_mode == "interactive":
        return "interactive"
    else:
        return "student"
```

### Agent Descriptions

| Agent | User Type | Mode | Approach | Max Tokens |
|-------|-----------|------|----------|-----------|
| **Conversational** | Any | N/A | Friendly, no tools | 800 |
| **Student** | Student | Standard | Fact-first synthesis with examples | 1500 |
| **Interactive Student** | Student | Interactive | Socratic method: hints, step-by-step | 1500 |
| **Teacher** | Teacher | N/A | Scholarly analysis, pedagogical insights | 3000 |

#### Agent Behavior Details

**Conversational Agent**
- No tools (no retrieval, no web search)
- Direct LLM response
- Examples: greeting, acknowledgment, casual chat

**Student Agent (Standard)**
- Uses Retrieval Tool + Web Search Tool
- Returns clear, example-rich explanations
- Focused on understanding
- System prompt emphasizes clarity and relevance

**Interactive Student Agent (Socratic)**
- Uses Retrieval Tool + Web Search Tool
- Poses guiding questions instead of direct answers
- Encourages critical thinking
- System prompt: "Ask leading questions, guide to discovery"

**Teacher Agent**
- Uses Retrieval Tool + Web Search Tool
- Provides scholarly insights, pedagogical analysis
- Discusses assessment strategies, common misconceptions
- System prompt: "Address educational challenges, suggest teaching methods"

---

## Message & Context Management

### AgentState Type Definition

The `AgentState` TypedDict contains all data passed between graph nodes:

```python
class AgentState(TypedDict, total=False):
    # Input (from API request)
    user_session_id: str
    user_id: str
    user_type: Literal["student", "teacher"]
    query: str
    language: str
    agent_mode: str              # "standard" or "interactive"
    student_grade: Literal["A", "B", "C", "D"]
    subjects: List[str]
    
    # Session Context
    session_metadata: SessionMetadata    # class_level, subject, chapter, lecture_id
    request_filters: Dict[str, Any]     # User-provided filters
    
    # Processing
    query_en: str                        # English translation
    detected_language: str               # Detected input language
    intent: QueryIntent                  # Parsed query intent
    query_type: str                      # "conversational" vs "curriculum_specific"
    documents: List[Document]            # Retrieved documents
    conversation_history: List[BaseMessage]  # Trimmed messages
    citations: List[Citation]            # Document citations for response
    timings: Dict[str, float]            # Performance metrics
    llm_calls: int                       # Counter of LLM invocations
    
    # Output
    response: str                        # Generated response
    final_language: str                  # Language for response
    is_translated: bool                  # Was translation needed?
```

### Conversation History Management

#### Maximum Message Turns

The system maintains conversation context via:

1. **Token-based trimming** (hard limit)
   - Max tokens: 2000 (configurable via `memory_token_limit`)
   - Strategy: Keep most recent messages, drop oldest first
   - Always starts on human (user) message

2. **Turn-based storage**
   - Redis buffer: last 30 turns
   - Memory context: trimmed to 2000 tokens
   - Full history: MongoDB for analytics

#### Example Flow

```
User Message 1: "Explain photosynthesis" (50 tokens)
Assistant Response 1: "Photosynthesis is..." (200 tokens)
User Message 2: "What about chlorophyll?" (30 tokens)
Assistant Response 2: "Chlorophyll absorbs..." (180 tokens)
...
(After 20 exchanges, ~1500 tokens accumulated)
User Message 21: "What about glucose?" (25 tokens)
→ TRIM: "Drop User Message 1 and Response 1" (250 tokens freed)
→ Total: ~1300 tokens
→ Continue...
```

### SessionMetadata Extraction

During **AnalyzeQueryNode**, metadata is extracted:

```python
class SessionMetadata(TypedDict, total=False):
    class_level: Optional[str]       # "Class 10", "Grade 12"
    subject: Optional[str]            # "Mathematics", "Biology"
    chapter: Optional[str]            # "Chapter 5", "Unit 2"
    lecture_id: Optional[str]         # Session ID
    last_topic: Optional[str]         # Recent topic summary
```

**Extraction Sources** (in priority order):
1. Current user query (explicit mention)
2. Conversation history (implicit context)
3. Request filters (user-provided)
4. Query classification LLM output

---

## Key Configuration Parameters

### Memory Configuration

```python
# From config.py
memory_buffer_size: int = 20              # Conversation turns to keep
memory_token_limit: int = 2000            # Max tokens for trimmed history
```

### Model Configuration

```python
model_name: str = "gpt-4o-mini"           # LLM model
embedding_model: str = "text-embedding-3-large"  # Embedding model
llm_temperature: float = 0.0              # Deterministic (no randomness)
```

### Retrieval Configuration

```python
retriever_top_k: int = 5                  # Documents to fetch
retriever_score_threshold: float = 0.4    # Min similarity score
# Hybrid search alpha: 0.8 (80% dense, 20% BM25 sparse)
```

### Token Limits

```python
max_tokens_brief: int = 800               # Quick answers
max_tokens_default: int = 1500            # Standard responses
max_tokens_detailed: int = 3000           # Deep dives
```

### Agent Configuration

```python
max_iterations: int = 5                   # ReAct agent loop max iterations
web_search_enabled: bool = True           # Enable/disable web search
```

### Example .env File

```env
# API Keys
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcak_...
PINECONE_INDEX=vidyaai-index

# Databases
MONGODB_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379

# Model Settings
MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.0

# Memory
MEMORY_BUFFER_SIZE=20
MEMORY_TOKEN_LIMIT=2000

# Retrieval
RETRIEVER_TOP_K=5
RETRIEVER_SCORE_THRESHOLD=0.4

# Agent
MAX_ITERATIONS=5
WEB_SEARCH_ENABLED=true
```

---

## Deployment & Running

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (recommended)
- OpenAI API key
- Pinecone API key

### Starting the Application

#### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build -d

# Services started:
# - API: http://localhost:8000
# - Redis: localhost:6379
# - MongoDB: localhost:27017
```

#### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Start MongoDB
mongod

# Start API server
python -m uvicorn main:app --reload
```

### Key Environment Files

- `.env` - Environment variables (required)
- `bm25_encoder.json` - Pre-trained BM25 encoder (included)
- `docker-compose.yml` - Service definitions

### Application Health

```bash
# Check API health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api

# Verify Redis
redis-cli ping

# Verify MongoDB
mongo --eval "db.adminCommand('ping')"
```

---

## Operational Insights for Maintainers

### Key Files to Know

| File | Purpose |
|------|---------|
| `config.py` | All configuration & settings |
| `main.py` | FastAPI app entry point |
| `graph.py` | LangGraph workflow definition |
| `state.py` | AgentState type definitions |
| `services/` | Business logic (memory, retrieval, validation, etc.) |
| `agents/` | Agent implementations |
| `nodes/` | Graph node implementations |
| `tools/` | ReAct agent tools |
| `models/` | Pydantic models (ChatSession, ChatMessage) |

### Critical Concepts

1. **Everything is async** - All operations use async/await patterns
2. **Token efficiency** - Token counting and trimming prevents budget overrun
3. **Dual storage** - Redis for speed, MongoDB for persistence
4. **LLM calls are expensive** - Summary and caching reduce repeated calls
5. **Context extraction** - Critical for filtering and retrieval accuracy
6. **Validation prevents hallucinations** - Groundedness checking is essential

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check state during execution
print(f"State query_type: {state.get('query_type')}")
print(f"Retrieved docs: {len(state.get('documents', []))}")

# Monitor Redis
redis-cli KEYS "chat:*"
redis-cli LRANGE "chat:{session_id}:buffer" 0 -1

# Monitor MongoDB
db.chatsessions.find({"session_id": session_id})
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "No documents found" | Retrieval threshold too high | Lower `retriever_score_threshold` |
| High latency | Large conversation history | Reduce `memory_buffer_size` |
| Hallucinations | Weak validation | Check `ResponseValidator` logic |
| Language not detected | Query too short | Add language hints in request |
| LLM rate limits | Too many requests | Implement backoff, increase caching TTL |

---

## Summary

VidyaAI-AGENT is a sophisticated educational chatbot that combines:

- **Smart routing**: Query classification determines the appropriate agent
- **Efficient memory**: Token-based trimming + Redis + MongoDB dual storage
- **Powerful retrieval**: Hybrid dense/sparse search from Pinecone
- **Validation**: Groundedness checking prevents hallucinations
- **Flexibility**: Multiple agents for different user types and pedagogical approaches
- **Performance**: Multi-layered caching and proactive document fetching

The system is designed to scale, maintain low latency, and provide accurate, contextually relevant educational responses.

---

## Contact & Support

For questions about this system, refer to the code comments and this guide. The codebase is well-documented with inline comments explaining complex logic.
