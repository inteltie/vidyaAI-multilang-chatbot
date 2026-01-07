# VidyaAI-AGENT: Technical Deep Dive

A detailed technical reference for developers and architects maintaining the project.

---

## 1. System Architecture Diagrams

### 1.1 Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Application                             │
│                          (main.py + BackendApp class)                        │
└─────────────────────────────────────────┬─────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
        ┌───────────▼──────────┐  ┌──────▼──────────┐  ┌────────▼────────┐
        │   Redis (aioredis)   │  │  MongoDB        │  │   Pinecone      │
        │                      │  │  (motor/beanie) │  │   (py-sdk)      │
        │  - Chat buffers      │  │                 │  │                 │
        │  - Web cache         │  │  - ChatSession  │  │  - Embeddings   │
        │  - Session data      │  │  - Messages     │  │  - Metadata     │
        │  - TTL: 1 hour       │  │  - Summaries    │  │  - Dense+Sparse │
        └──────────────────────┘  └─────────────────┘  └─────────────────┘
                    │
        ┌───────────▼────────────────────────────────────────┐
        │       LangGraph Compiled StateGraph                │
        │      (ChatbotGraphBuilder.build().compile())       │
        │                                                     │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  Nodes (Linear + Conditional Routing)       │  │
        │  │                                              │  │
        │  │  1. load_memory (LoadMemoryNode)            │  │
        │  │     ├─ Load from Redis/MongoDB              │  │
        │  │     └─ Return: session, buffer, summary     │  │
        │  │                                              │  │
        │  │  2. analyze_query (AnalyzeQueryNode)        │  │
        │  │     ├─ Language detection → English         │  │
        │  │     ├─ Query classification (LLM)           │  │
        │  │     ├─ Context extraction                   │  │
        │  │     ├─ Proactive RAG fetch                  │  │
        │  │     └─ Return: query_type, documents       │  │
        │  │                                              │  │
        │  │  3. Router: _route_to_agent()               │  │
        │  │     ├─ IF conversational → Agent            │  │
        │  │     └─ IF educational → Route by user_type  │  │
        │  │                                              │  │
        │  │  4. Agent Nodes (Educational)               │  │
        │  │     ├─ student_agent (ReAct with tools)     │  │
        │  │     ├─ interactive_student_agent (Socratic) │  │
        │  │     └─ teacher_agent (Scholarly)            │  │
        │  │                                              │  │
        │  │  5. Agent Node (Conversational)             │  │
        │  │     └─ Direct LLM response (no tools)       │  │
        │  │                                              │  │
        │  │  6. groundedness_check (Validation)         │  │
        │  │     ├─ Is response grounded?                │  │
        │  │     ├─ Intent alignment check               │  │
        │  │     ├─ Ambiguity detection                  │  │
        │  │     └─ Max 1 retry on failure               │  │
        │  │                                              │  │
        │  │  7. translate_response (Language)           │  │
        │  │     └─ Translate to user's language         │  │
        │  │                                              │  │
        │  │  8. save_memory (Persistence)               │  │
        │  │     ├─ Redis: immediate write               │  │
        │  │     └─ MongoDB: async background            │  │
        │  │                                              │  │
        │  └─────────────────────────────────────────────┘  │
        │                                                     │
        └─────────────────────────────────────────────────────┘
                    │
        ┌───────────▼────────────────────────────────────────┐
        │          External LLM & Tools                       │
        │                                                     │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  OpenAI API Calls                           │  │
        │  │  ├─ Chat Completions (gpt-4o-mini)         │  │
        │  │  ├─ Embeddings (text-embedding-3-large)    │  │
        │  │  └─ Web Search (integrated in ChatGPT)      │  │
        │  └─────────────────────────────────────────────┘  │
        │                                                     │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  Tools (Inside Agents)                      │  │
        │  │  ├─ RetrievalTool (Pinecone hybrid search)  │  │
        │  │  └─ WebSearchTool (OpenAI web search)       │  │
        │  └─────────────────────────────────────────────┘  │
        │                                                     │
        └─────────────────────────────────────────────────────┘
```

### 1.2 Request Processing Pipeline

```
HTTP POST /api/chat
  │
  ├─→ [Validate] ChatRequest model
  │     └─ user_id, session_id, query, language, user_type, agent_mode
  │
  ├─→ [Initialize] AgentState
  │     ├─ Input fields: user_id, query, user_type, etc.
  │     ├─ Processing fields: empty (filled by nodes)
  │     └─ Output fields: empty (filled by agents)
  │
  ├─→ [Graph Execution] graph.invoke(state)
  │     │
  │     ├─→ Node: load_memory
  │     │     └─ Returns: (session, buffer, summary) tuple
  │     │
  │     ├─→ Node: analyze_query
  │     │     ├─ LLM Call #1: Language detection + classification
  │     │     ├─ LLM Call #2: Retrieve documents proactively
  │     │     └─ Returns: query_type, translated_query, documents, metadata
  │     │
  │     ├─→ [Conditional] Router: route_to_agent()
  │     │     └─ Branch point based on query_type
  │     │
  │     ├─→ [Branch A] IF conversational:
  │     │     ├─→ Node: conversational_agent
  │     │     │     └─ LLM Call #3: Simple friendly response
  │     │     │
  │     │     └─→ Node: translate_response
  │     │         └─ LLM Call #4 (if needed): Translate to user language
  │     │
  │     ├─→ [Branch B] IF educational (Route by user_type):
  │     │     │
  │     │     ├─→ Student Path:
  │     │     │     ├─→ Node: student_agent
  │     │     │     │     ├─ LLM Call #3: ReAct loop (up to 5 iterations)
  │     │     │     │     │     ├─ Call Retrieval Tool (doc search)
  │     │     │     │     │     ├─ Call Web Search Tool (if needed)
  │     │     │     │     │     └─ Synthesize response
  │     │     │     │     └─ Returns: response (gpt-4o-mini)
  │     │     │     │
  │     │     │     ├─→ Node: groundedness_check
  │     │     │     │     └─ LLM Call #4: Validate against documents
  │     │     │     │
  │     │     │     └─→ Node: translate_response
  │     │     │         └─ LLM Call #5 (if needed)
  │     │     │
  │     │     ├─→ Interactive Student Path: (same as above)
  │     │     │
  │     │     └─→ Teacher Path: (same as above)
  │     │
  │     └─→ Node: save_memory
  │           ├─ Write to Redis (immediate)
  │           └─ Async task: Write to MongoDB + update summary
  │
  └─→ [Response] ChatResponse
        ├─ response: Final answer
        ├─ language: Language of response
        ├─ citations: Document references
        ├─ session_id: For frontend continuity
        └─ metadata: Query analysis results
```

### 1.3 Data Flow Through Services

```
ChatRequest
    │
    ├─→ QueryClassifier (LLM)
    │     └─ Returns: QueryClassification
    │         ├─ query_type
    │         ├─ translated_query
    │         ├─ subjects
    │         ├─ class_level
    │         └─ lecture_id
    │
    ├─→ RetrieverService (Pinecone)
    │     ├─ Embed query (OpenAI)
    │     ├─ BM25 encode (sparse)
    │     ├─ Hybrid search (alpha=0.8)
    │     └─ Returns: List[Document]
    │
    ├─→ [ReAct Loop - Agent Processing]
    │     │
    │     ├─→ Tool: Retrieval
    │     │     └─ Returns: formatted documents
    │     │
    │     ├─→ Tool: Web Search
    │     │     ├─ Cache lookup (Redis)
    │     │     ├─ OpenAI chat (if not cached)
    │     │     └─ Returns: web summary
    │     │
    │     └─→ LLM: Generate response
    │           └─ Returns: response string
    │
    ├─→ ResponseValidator (LLM)
    │     ├─ Check groundedness
    │     ├─ Check intent alignment
    │     ├─ Detect ambiguity
    │     └─ Returns: ValidationResult
    │           ├─ is_valid: bool
    │           ├─ needs_clarification: bool
    │           └─ feedback: str
    │
    ├─→ Translator (LLM, if needed)
    │     └─ Translate to user_language
    │
    └─→ MemoryService (Redis + MongoDB)
          ├─ Add to Redis buffer
          └─ Async save to MongoDB
```

---

## 2. Node-by-Node Breakdown

### 2.1 LoadMemoryNode

**File**: `nodes/load_memory.py`

**Purpose**: Retrieve conversation context from persistent storage

**Inputs from State**:
- `user_session_id`
- `user_id`

**Process**:
```python
async def __call__(state: AgentState) -> AgentState:
    session_id = state["user_session_id"]
    user_id = state["user_id"]
    
    # Ensure session exists, get summary + buffer
    session, buffer, summary = await memory_service.ensure_session(user_id, session_id)
    
    # Get context (token-trimmed messages)
    summary, trimmed_history = await memory_service.get_context(session_id)
    
    # Update state
    state["conversation_history"] = trimmed_history
    state["session_metadata"] = extract_metadata_from_session(session)
    
    return state
```

**Outputs to State**:
- `conversation_history`: List[BaseMessage] (trimmed to 2000 tokens)
- `session_metadata`: SessionMetadata object

**Key Logic**:
1. Check Redis first (hot cache, 1-hour TTL)
2. If not in Redis, load from MongoDB
3. Rebuild Redis buffer if empty
4. Trim messages to token limit (keep recent)
5. Return summary separately (doesn't count tokens)

---

### 2.2 AnalyzeQueryNode (MERGED STEP)

**File**: `nodes/analyze_query.py`

**Purpose**: Translate query, classify type, extract context, and proactively fetch documents

**Inputs from State**:
- `query` (original, possibly non-English)
- `language` (optional, detected if not provided)
- `conversation_history` (for context)

**Process**:

```python
async def __call__(state: AgentState) -> AgentState:
    query = state["query"]
    
    # 1. CLASSIFY & TRANSLATE
    classification = await query_classifier.analyze(
        query=query,
        history=state["conversation_history"]
    )
    # Returns: QueryClassification with:
    #   - query_type: "conversational" | "curriculum_specific"
    #   - translated_query: English version
    #   - subjects: ["Math", "Science", ...]
    #   - class_level, chapter, lecture_id (if detected)
    
    # 2. EXTRACT CONTEXT
    session_metadata = {
        "class_level": classification.class_level,
        "subject": classification.extracted_subject,
        "chapter": classification.chapter,
        "lecture_id": classification.lecture_id,
    }
    
    # 3. PROACTIVE RAG FETCH (Optimization)
    if classification.query_type == "curriculum_specific":
        # Merge request filters with extracted metadata
        filters = {**state.get("request_filters", {}), **session_metadata}
        
        documents = await retriever.retrieve(
            query_en=classification.translated_query,
            filters=filters,
            intent=classify_intent(classification.subjects)
        )
    else:
        documents = []
    
    # 4. UPDATE STATE
    state["query_en"] = classification.translated_query
    state["query_type"] = classification.query_type
    state["detected_language"] = detected_language
    state["documents"] = documents
    state["session_metadata"] = session_metadata
    state["subjects"] = classification.subjects
    
    return state
```

**Outputs to State**:
- `query_en`: Translated query in English
- `query_type`: "conversational" | "curriculum_specific"
- `documents`: List[Document] (from Pinecone if educational)
- `session_metadata`: Extracted class_level, subject, chapter, lecture_id
- `detected_language`: Original language code (en, es, hi, etc.)
- `subjects`: List of detected subjects

**LLM Calls**: 1 (QueryClassifier with structured output)

**Performance**:
- Language detection: 100ms (LLM)
- Document retrieval: 1000ms (Pinecone + embedding)
- Total: ~1100ms

---

### 2.3 Routing Nodes

**Files**: `graph.py` (routing logic)

#### Route #1: Agent Type Selection

```python
def _route_to_agent(state: AgentState) -> Literal["conversational", "educational"]:
    query_type = state.get("query_type", "curriculum_specific")
    
    if query_type == "conversational":
        return "conversational"  # → conversational_agent node
    else:
        return "educational"     # → route_educational_user (pass-through)
```

#### Route #2: Educational User Type Selection

```python
def _route_educational_user(state: AgentState) -> Literal["student", "interactive", "teacher"]:
    user_type = state.get("user_type", "student")
    agent_mode = state.get("agent_mode", "standard")
    
    if user_type == "teacher":
        return "teacher"         # → teacher_agent node
    elif agent_mode == "interactive":
        return "interactive"     # → interactive_student_agent node
    else:
        return "student"         # → student_agent node (default)
```

---

### 2.4 Agent Nodes

**Files**: `agents/*.py` and `nodes/*_agent_node.py`

All agent nodes share the same signature:

```python
async def __call__(state: AgentState) -> AgentState:
    # 1. Build system prompt (tailored to agent type)
    system_prompt = AGENT_SYSTEM_PROMPTS[agent_type]
    
    # 2. Format context
    conversation_context = format_conversation(state["conversation_history"])
    summary_context = state.get("summary", "")
    documents_context = format_documents(state["documents"])
    
    # 3. Create full prompt
    full_prompt = f"""
{system_prompt}

Summary of conversation: {summary_context}

Recent messages:
{conversation_context}

Retrieved documents:
{documents_context}

User's question: {state["query_en"]}

[Agent should now respond using available tools if educational]
"""
    
    # 4. For educational agents: Use ReAct with tools
    if agent_type in ["student", "interactive", "teacher"]:
        response = await react_agent.invoke({
            "input": full_prompt,
            "tools": [retrieval_tool, web_search_tool],
            "max_iterations": settings.max_iterations
        })
    
    # 5. For conversational: Direct LLM response
    else:
        response = await llm.ainvoke(full_prompt, max_tokens=800)
    
    state["response"] = response
    state["llm_calls"] += 1
    
    return state
```

#### Agent System Prompts

```
StudentAgent:
"You are an educational tutor for Indian students. Explain concepts clearly
with examples. Use the retrieved documents as your primary source. If documents
are insufficient, use web search. Keep responses concise but comprehensive.
Focus on understanding, not just facts."

InteractiveStudentAgent:
"You are a Socratic tutor. Instead of directly answering, ask guiding questions
to help students discover the answer themselves. First, ask what they know.
Then, suggest connections. Finally, ask what they can conclude."

TeacherAgent:
"You are an expert educational consultant. Provide scholarly analysis including:
1. Key concepts explained
2. Common student misconceptions
3. Suggested teaching strategies
4. Assessment ideas
Focus on pedagogy, not just content."

ConversationalAgent:
"You are a friendly educational assistant. Respond warmly to greetings,
acknowledge thanks, and provide helpful redirects. Keep responses brief and friendly."
```

#### ReAct Agent Loop

```python
# For educational agents only
for iteration in range(max_iterations):
    # 1. Get LLM response with tool use
    response = await llm.ainvoke(
        prompt + previous_observations,
        tools=[retrieval_tool, web_search_tool]
    )
    
    # 2. If tool_calls present, execute
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.name == "retrieve_documents":
                docs = await retrieval_tool.execute(
                    query=tool_call.args["query"],
                    filters=state["session_metadata"]
                )
            elif tool_call.name == "web_search":
                results = await web_search_tool.execute(
                    query=tool_call.args["query"]
                )
            
            observation = format_tool_output(docs or results)
            state["documents"] = docs if docs else state["documents"]
    
    # 3. If no more tool calls, agent is done
    else:
        break
    
    iteration += 1
```

---

### 2.5 GroundednessCheckNode (Validation)

**File**: `nodes/groundedness_check.py`

**Purpose**: Validate response is grounded in documents and handles ambiguity

**Inputs from State**:
- `response`: Generated response string
- `query_en`: Original query in English
- `documents`: Retrieved documents
- `subjects`: Detected subjects

**Process**:

```python
async def __call__(state: AgentState) -> AgentState:
    response = state["response"]
    documents = state["documents"]
    query = state["query_en"]
    subjects = state["subjects"]
    
    # 1. VALIDATE
    validation = await response_validator.validate(
        query=query,
        response=response,
        documents=documents,
        intent_subjects=subjects
    )
    
    # 2. HANDLE RESULTS
    if validation.needs_clarification:
        # User is confused, ask which interpretation
        state["clarification_message"] = validation.clarification_question
        state["response"] = validation.clarification_question
        state["validation_results"] = validation
        return state
    
    if not validation.is_valid and not state.get("is_correction"):
        # Response is hallucinated, retry once
        state["is_correction"] = True
        
        # Call agent again with feedback
        feedback = f"Your response contains issues: {validation.feedback}"
        state["response"] = ""  # Clear response
        
        # Re-invoke agent with feedback
        updated_state = await agent_node(state, feedback)
        
        state["response"] = updated_state["response"]
        state["validation_results"] = validation
    
    else:
        # Response is valid, proceed
        state["validation_results"] = validation
    
    return state
```

**ValidationResult**:
```python
{
    "is_valid": True/False,
    "needs_clarification": True/False,
    "reasoning": "explanation",
    "feedback": "optional corrections",
    "clarification_question": "Which interpretation?"
}
```

**Max Retries**: 1 (prevent infinite loops)

---

### 2.6 TranslateResponseNode

**File**: `nodes/translate_response.py`

**Purpose**: Translate response back to user's original language if needed

**Inputs from State**:
- `response`: Generated response (in English)
- `detected_language`: User's original language
- `query_en`: English query (for context)

**Process**:

```python
async def __call__(state: AgentState) -> AgentState:
    response = state["response"]
    user_language = state.get("detected_language", "en")
    
    # 1. SKIP IF ALREADY IN USER'S LANGUAGE
    if user_language == "en":
        state["is_translated"] = False
        state["final_language"] = "en"
        return state
    
    # 2. CHECK CACHE
    cache_key = CacheService.generate_key("translation", 
                                         f"{response[:50]}_{user_language}")
    cached = await CacheService.get(cache_key)
    if cached:
        state["response"] = cached
        state["is_translated"] = True
        state["final_language"] = user_language
        return state
    
    # 3. TRANSLATE
    prompt = f"""Translate this educational response to {user_language}.
    Keep the technical accuracy and structure intact.
    
    Original: {response}
    
    Translated:"""
    
    translated = await translator.translate(
        text=response,
        source_language="en",
        target_language=user_language
    )
    
    # 4. CACHE TRANSLATION
    await CacheService.set(cache_key, translated, ttl=86400)
    
    # 5. UPDATE STATE
    state["response"] = translated
    state["is_translated"] = True
    state["final_language"] = user_language
    
    return state
```

**Supported Languages**: en, es, hi, te, ta, kn, ml, etc. (configurable)

---

### 2.7 SaveMemoryNode

**File**: `nodes/save_memory.py`

**Purpose**: Persist conversation to Redis and MongoDB

**Inputs from State**:
- `response`: Final response to save
- `query`: Original user query
- `user_session_id`: Session ID
- `user_id`: User ID

**Process**:

```python
async def __call__(state: AgentState) -> AgentState:
    session_id = state["user_session_id"]
    user_id = state["user_id"]
    query = state["query"]
    response = state["response"]
    
    # 1. SAVE TO REDIS (Immediate)
    await memory_service.add_message(session_id, "user", query)
    await memory_service.add_message(session_id, "assistant", response)
    
    # 2. BACKGROUND SAVE TO MONGODB (Async)
    asyncio.create_task(
        memory_service.background_save_message(session_id, user_id, "user", query)
    )
    asyncio.create_task(
        memory_service.background_save_message(session_id, user_id, "assistant", response)
    )
    
    # 3. SUMMARY UPDATE (Every 20 messages)
    # Triggered inside background_save_message
    
    return state
```

**Storage Architecture**:

```
Redis (Hot Cache):
  Key: chat:{session_id}:buffer
  Value: List of {role, content} JSON objects
  TTL: 1 hour
  Size: Last 30 messages

MongoDB (Cold Storage):
  Collection: chatsessions
  Document:
  {
    "session_id": "session_123",
    "user_id": "user_456",
    "messages": [
      {"role": "user", "text": "...", "timestamp": ...},
      {"role": "assistant", "text": "...", "timestamp": ...},
      ...
    ],
    "summary": "User asked about...",
    "created_at": "...",
    "last_updated": "..."
  }
```

---

## 3. Query Classification Deep Dive

### 3.1 Classification Flow

```
Input: Query + Conversation History
    │
    ├─ Step 1: Heuristics Check (Optional)
    │   ├─ IF "hi", "hello", "thanks" → conversational (NO LLM)
    │   └─ Else continue to Step 2
    │
    ├─ Step 2: LLM Classification
    │   ├─ Prompt includes:
    │   │   ├─ Last 4 conversation turns (context)
    │   │   ├─ Current query
    │   │   └─ Classification criteria
    │   │
    │   ├─ LLM decides:
    │   │   ├─ query_type: conversational | curriculum_specific
    │   │   ├─ confidence: 0.0-1.0
    │   │   └─ subjects: [list of subjects]
    │   │
    │   └─ SIMULTANEOUSLY extracts:
    │       ├─ class_level: "Class 10", "Grade 12", etc.
    │       ├─ extracted_subject: "Algebra", "Organic Chemistry"
    │       ├─ chapter: "Quadratic Equations"
    │       └─ lecture_id: "42", "session_10"
    │
    └─ Output: QueryClassification object
```

### 3.2 Classification Rules

```
CONVERSATIONAL (No RAG needed):
  - Greeting: "hi", "hello", "hey"
  - Expression of gratitude: "thanks", "thank you"
  - Acknowledgment: "ok", "got it", "cool"
  - Goodbye: "bye", "goodbye", "see ya"
  - Who are you: "who are you", "what are you"
  - Vague help: "i need help" (without specific topic)

CURRICULUM_SPECIFIC (RAG required):
  - DEFAULT for anything educational
  - Examples:
    ├─ "Explain photosynthesis"
    ├─ "What is the capital of France?"
    ├─ "How do I solve quadratic equations?"
    ├─ "What are the symptoms of malaria?"
    ├─ "Describe the French Revolution"
    └─ "Calculate the area of a circle"
  - RULE: If in doubt, choose curriculum_specific (safe default)
```

### 3.3 Subject Detection

The system detects one or more subjects from a predefined list:

```python
AVAILABLE_SUBJECTS = [
    "Math",           # Algebra, Geometry, Calculus, etc.
    "Science",        # Physics, Chemistry, Biology
    "History",        # World history, Indian history
    "Geography",      # Physical, human, economic
    "English",        # Literature, grammar
    "Social Studies", # Civics, economics
    "General"         # Fallback for unknown/mixed
]
```

**Detection Method**:
1. Query contains explicit mention: "math problem" → Math
2. Context from history: "We were doing chemistry..." → Science
3. LLM inference: "mitochondria" → Science
4. Multiple subjects: "Compare India's economy to China's" → [History, Geography]
5. Fallback: "General" if nothing matches

---

## 4. Retrieval & Hybrid Search Internals

### 4.1 Hybrid Search Process

```
Query: "Explain photosynthesis"
    │
    ├─ Step 1: EMBEDDING (Dense Vector)
    │   ├─ Model: text-embedding-3-large
    │   ├─ Dimensions: 1536
    │   ├─ Process: Convert query to vector
    │   └─ Output: dense = [0.2, -0.3, 0.1, ..., 0.15] (1536 values)
    │
    ├─ Step 2: BM25 ENCODING (Sparse Vector)
    │   ├─ Model: Pre-trained BM25Encoder
    │   ├─ Source: bm25_encoder.json (loaded at startup)
    │   ├─ Process: TF-IDF based keyword matching
    │   └─ Output: sparse = {
    │               "indices": [12, 45, 127, 234],
    │               "values": [0.8, 0.6, 0.4, 0.3]
    │             }
    │
    ├─ Step 3: HYBRID SCALING (Alpha = 0.8)
    │   ├─ Formula: score = 0.8 * dense + 0.2 * sparse
    │   ├─ Process:
    │   │   ├─ Multiply dense vector by 0.8
    │   │   ├─ Multiply sparse values by 0.2
    │   │   └─ Keep indices same for sparse
    │   │
    │   └─ Output:
    │       ├─ scaled_dense = [0.16, -0.24, 0.08, ..., 0.12]
    │       └─ scaled_sparse = {
    │           "indices": [12, 45, 127, 234],
    │           "values": [0.16, 0.12, 0.08, 0.06]
    │         }
    │
    ├─ Step 4: FILTER CONSTRUCTION
    │   ├─ From request: class_level="Class 10"
    │   ├─ From analysis: subject="Biology", chapter="Plant Processes"
    │   └─ Output: Pinecone filter = {
    │       "class_level": {"$eq": "Class 10"},
    │       "subject": {"$eq": "Biology"},
    │       "chapter": {"$eq": "Plant Processes"}
    │     }
    │
    ├─ Step 5: PINECONE QUERY
    │   └─ query(
    │       vector=scaled_dense,           # 1536-dim
    │       sparse_vector=scaled_sparse,   # {"indices": [...], "values": [...]}
    │       filter={...},                  # Metadata filters
    │       top_k=5,                       # Top 5 results
    │       include_metadata=True          # Include metadata in response
    │     )
    │
    └─ Step 6: POST-PROCESS RESULTS
        ├─ Filter by score threshold (0.4)
        ├─ Format documents with metadata
        └─ Output: List[Document]
```

### 4.2 Document Format

```python
{
    "id": "chunk_12345",
    "score": 0.87,                    # Hybrid score
    "text": "Photosynthesis is the process by which plants...",
    "metadata": {
        "lecture_id": 42,
        "transcript_id": 100,
        "chunk_id": "chunk_12345",
        "subject": "Biology",
        "subject_id": 5,
        "chapter": "Plant Processes",
        "topics": "photosynthesis, chlorophyll, glucose",
        "class_name": "Class 10",
        "class_id": 10,
        "teacher_name": "Dr. Sharma",
        "teacher_id": 23
    }
}
```

### 4.3 Filter Enforcement

**CRITICAL**: Filters from user request are **strictly enforced** and LLM-extracted filters are **ignored**.

```python
# From retrieval_tool.py
async def execute(self, query: str, filters: Dict = None, **kwargs):
    # filters come from request (user-provided)
    # kwargs come from LLM (ignored for filtering)
    
    if kwargs:
        logger.info("Ignoring LLM-extracted filters: %s", kwargs)
    
    # Use ONLY filters from user
    docs = await retriever.retrieve(
        query_en=query,
        filters=filters,  # ← Strictly user-provided
        intent=QueryIntent.CONCEPT_EXPLANATION
    )
```

**Rationale**: Prevents LLM from overriding user's explicit filter choices.

---

## 5. Memory & Caching Strategy

### 5.1 Token Trimming Algorithm

```
Original messages (token counts):
  1. User: "What is photosynthesis?" (10 tokens)
  2. Assistant: "Photosynthesis is..." (150 tokens)
  3. User: "Tell me more about chlorophyll" (12 tokens)
  4. Assistant: "Chlorophyll is..." (180 tokens)
  5. User: "And glucose?" (5 tokens)
  6. Assistant: "Glucose is produced..." (140 tokens)
  Total: ~500 tokens

After many exchanges (1900 tokens total):
  ├─ Summary available: 50 tokens (counts separate, not included)
  └─ Trimming needed to fit in 2000 token limit

Trimming Process (max_tokens=2000, strategy="last"):
  1. Start from MOST RECENT
  2. Keep messages working backward
  3. Drop oldest messages first
  4. Stop when total <= 2000 tokens
  5. Always start on "human" (user message)

Result (trimmed to 1800 tokens):
  Messages 3, 4, 5, 6 kept (~500 tokens)
  Message 2 dropped (exceeds limit)
  Message 1 dropped
  → Context window has space for summary + new query + response
```

### 5.2 Summary Generation

```
Trigger: Every 20 new messages

Process:
  1. Get previous summary (if exists)
  2. Fetch last 20 messages
  3. Create prompt:
     "Here's what we've discussed: {prev_summary}
      Latest messages: {recent_20}
      Please update the summary..."
  
  4. LLM generates new summary (1-2 sentences)
  5. Store in MongoDB ChatSession.summary
  6. Use in next conversation (doesn't count tokens)

Example:
  Old Summary: "User asked about photosynthesis and learned about chlorophyll."
  New Messages: [User: "What about glucose?", Assistant: "Glucose is...", ...]
  New Summary: "User learning about photosynthesis including chlorophyll, 
               glucose production, and energy transfer in plants."
```

### 5.3 Redis Buffer Management

```
Redis Key: "chat:{session_id}:buffer"
Type: List (LPUSH / RPUSH)
TTL: 1 hour (auto-expires)

Operations:
  1. RPUSH (add to end): new message
  2. LTRIM (trim from start): keep last 30 turns
  3. EXPIRE: set 1-hour TTL on key
  4. LRANGE: retrieve all messages

Size Control:
  - Max 30 turns (≈ 3000-5000 tokens)
  - Older turns auto-expire after 1 hour
  - MongoDB backup ensures no data loss

Why Redis?
  - O(1) access for most recent messages
  - Fast JSON serialization
  - Built-in TTL (auto-cleanup)
  - Perfect for active sessions
```

---

## 6. Tool Execution & ReAct Loop

### 6.1 Tool Invocation Flow

```
Agent LLM Call (with tools available):
    │
    ├─ LLM receives:
    │   ├─ System prompt
    │   ├─ Message history
    │   ├─ Tool descriptions (schema)
    │   └─ Instruction: "Use tools if needed"
    │
    ├─ LLM Response (2 options):
    │   │
    │   ├─ Option A: Tool Calls
    │   │   └─ {
    │   │       "tool_calls": [
    │   │         {
    │   │           "id": "call_123",
    │   │           "name": "retrieve_documents",
    │   │           "args": {"query": "photosynthesis"}
    │   │         }
    │   │       ]
    │   │     }
    │   │
    │   └─ Option B: Final Response
    │       └─ {
    │           "content": "Based on the documents, photosynthesis is..."
    │         }
    │
    ├─ Agent checks response:
    │   └─ IF tool_calls present → Execute tools
    │       └─ Iterate (max 5 times)
    │       
    └─ END: Return final response (Option B)
```

### 6.2 ReAct (Reasoning + Acting) Loop

```
Iteration 1:
  ├─ LLM Thought: "I need to search for photosynthesis"
  ├─ Action: retrieve_documents(query="photosynthesis")
  ├─ Tool Execution:
  │   ├─ Embed query
  │   ├─ BM25 encode
  │   ├─ Hybrid search Pinecone
  │   └─ Return 5 documents
  ├─ Observation: [Document 1, Document 2, ...]
  └─ Add to context for next call

Iteration 2:
  ├─ LLM Thought: "Docs explain photosynthesis. Need chlorophyll detail"
  ├─ Action: web_search(query="chlorophyll role in photosynthesis")
  ├─ Tool Execution:
  │   ├─ Check Redis cache (HIT)
  │   └─ Return cached result
  ├─ Observation: "Web search result about chlorophyll..."
  └─ Add to context for next call

Iteration 3:
  ├─ LLM Thought: "I have enough info. Will write response"
  ├─ Action: (None - ready to respond)
  ├─ Response: "Photosynthesis is...
               [synthesized from documents + web search]"
  └─ DONE (no tool_calls)

Max Iterations: 5 (prevents infinite loops)
```

### 6.3 Tool Parameters

#### RetrievalTool

```python
{
    "query": "photosynthesis",              # Required
    "class_name": "Class 10",               # Optional
    "subject": "Biology",                   # Optional
    "chapter": "Plant Processes",           # Optional
    "topics": "chlorophyll, glucose",       # Optional
    "lecture_id": 42,                       # Optional
    "trainer_name": "Dr. Sharma",           # Optional
    "class_id": 10,                         # Optional
    "subject_id": 5                         # Optional
}
```

#### WebSearchTool

```python
{
    "query": "photosynthesis 2024 research"  # Required
}
```

---

## 7. Response Validation Logic

### 7.1 Validation Process

```
Response Validation:
    │
    ├─ Input:
    │   ├─ User query
    │   ├─ Agent response
    │   ├─ Retrieved documents
    │   └─ Detected subjects
    │
    ├─ Step 1: GROUNDEDNESS CHECK
    │   ├─ Is response supported by documents?
    │   ├─ LLM compares response against doc text
    │   └─ Score: grounded or hallucinated?
    │
    ├─ Step 2: INTENT ALIGNMENT CHECK
    │   ├─ Query detected subject: Biology
    │   ├─ Response topic: Photosynthesis (Biology) ✓
    │   └─ Are they aligned?
    │
    ├─ Step 3: AMBIGUITY DETECTION
    │   ├─ Are multiple valid interpretations?
    │   ├─ Example: "Transformers" (electrical vs AI)
    │   ├─ Did agent pick one or stay vague?
    │   └─ Does user need clarification?
    │
    └─ Output: ValidationResult
        ├─ is_valid: True/False
        ├─ needs_clarification: True/False
        ├─ feedback: "correction if invalid"
        └─ clarification_question: "Which one did you mean?"
```

### 7.2 Validation Rules

```python
# Rule 1: Groundedness
if response_claims_not_in_documents:
    is_valid = False
    feedback = "Your answer mentions facts not found in documents"
else:
    is_valid = True

# Rule 2: Intent Alignment
if response_subject != detected_subject:
    # Different subject detected
    if has_strong_evidence_for_multiple:
        needs_clarification = True
        clarification_question = "Did you mean...?"
    else:
        is_valid = False
        feedback = "Response doesn't address the right topic"

# Rule 3: Max Retries
if not is_valid and is_correction:
    # Already retried once, don't retry again
    is_valid = True  # Accept as-is to avoid infinite loops
```

### 7.3 Correction Flow

```
Agent generates response
    │
    ├─ Validation Check:
    │   ├─ is_valid = False
    │   ├─ needs_clarification = False
    │   └─ feedback = "Response has hallucinations"
    │
    ├─ Correction Retry (only 1x):
    │   ├─ Add feedback to prompt
    │   ├─ Re-invoke agent
    │   └─ Get new response
    │
    ├─ Validation Again:
    │   └─ IF still invalid → Accept (prevent infinite loop)
    │
    └─ Continue to translation
```

---

## 8. Distributed System Considerations

### 8.1 Concurrency Model

```
All operations are ASYNC:
  - FastAPI handles multiple requests concurrently
  - Each request runs graph.invoke() independently
  - No global state (stateless design)
  
Example:
  User A sends query → loads memory for A → processes
  User B sends query → loads memory for B → processes
  (Both run concurrently, no interference)
```

### 8.2 Database Consistency

```
Redis (Session Buffer):
  - Single source of truth for active conversation
  - TTL auto-expires stale sessions
  - No synchronization issues (single writer)

MongoDB (Persistent):
  - Background async save (eventual consistency OK)
  - Indexes on session_id for fast lookup
  - No race conditions (Beanie handles this)

Pinecone (Vector DB):
  - Read-only for chatbot (no writes)
  - Indexed for fast hybrid search
  - Pre-indexed with documents (offline)
```

### 8.3 Rate Limiting

Not explicitly implemented, but considerations:

```python
# OpenAI API Rate Limits (handled by client library):
#   - gpt-4o-mini: 200 requests/min, 4M tokens/min
#   - text-embedding-3-large: 3000 requests/min
#   - Web search: included in chat completions

# Strategies to handle limits:
#   1. Cache aggressively (Redis)
#   2. Batch requests if possible
#   3. Use exponential backoff on errors
#   4. Monitor token usage via LangChain callbacks
```

---

## 9. Error Handling & Fallbacks

### 9.1 Error Handling Pattern

```python
# Pattern used throughout codebase:
try:
    result = await operation()
except SpecificError as e:
    logger.error("Operation failed: %s", e)
    return fallback_result()

# Examples:

# 1. LLM Call Fails
try:
    classification = await llm.ainvoke(prompt)
except Exception as e:
    logger.warning("Classification failed, defaulting to curriculum_specific")
    return QueryClassification(
        query_type="curriculum_specific",
        translated_query=query,
        confidence=0.0
    )

# 2. Retrieval Fails
try:
    docs = await retriever.retrieve(query)
except Exception as e:
    logger.error("Retrieval failed: %s", e)
    return []  # Empty list → web search fallback

# 3. Translation Fails
try:
    translated = await translator.translate(response, target_language)
except Exception as e:
    logger.warning("Translation failed, returning original")
    return response  # Return English original

# 4. Database Save Fails
try:
    await session.add_message(role, content)
except Exception as e:
    logger.error("MongoDB save failed: %s", e)
    # Message still in Redis, will retry in background
```

### 9.2 Fallback Chains

```
Scenario: User asks "Latest AI developments"
    │
    ├─ Try: Retrieval (curriculum docs)
    │   └─ Result: Empty (topic too recent)
    │
    ├─ Try: Web Search
    │   └─ Result: Found 5 relevant articles
    │
    └─ Outcome: Use web search results + synthesize response

Scenario: Translation service down
    │
    ├─ Try: Translator.translate()
    │   └─ Fails with timeout
    │
    ├─ Try: Return original English response
    │   └─ Success (user may understand English)
    │
    └─ Outcome: User gets answer in English instead of native language
```

---

## 10. Performance Optimization Techniques

### 10.1 Token Efficiency

**Problem**: Token usage = cost + latency

**Solutions Implemented**:
```
1. Summaries (large context w/o tokens):
   - Store summary in MongoDB
   - Use in prompt (doesn't count tokens)
   - Update every 20 messages

2. Token Trimming:
   - Keep max 2000 tokens in history
   - Drop oldest messages first
   - Keep recent context

3. Proactive RAG Fetch:
   - Fetch documents during AnalyzeQuery
   - Reuse in agent prompt
   - Avoid redundant searches

4. Smart Prompting:
   - System prompt is general (cached in model weights)
   - Context packed efficiently
   - Avoid repeating info
```

### 10.2 Latency Reduction

```
Bottlenecks:
  1. Embedding generation: ~500ms
  2. Pinecone query: ~1000ms
  3. LLM inference: ~1-3 seconds
  4. Tool execution: ~500-2000ms

Optimizations:
  1. Parallel loading:
     - Load memory AND analyze query simultaneously?
     - NO - need memory for history
     - But fetch docs during analysis (proactive)
  
  2. Caching:
     - Embeddings: precomputed in Pinecone
     - Web search: 24-hour Redis cache
     - Translations: 24-hour Redis cache
     - Session buffer: 1-hour Redis cache
  
  3. Connection pooling:
     - Redis: aioredis handles pooling
     - MongoDB: Motor handles pooling
     - Pinecone: SDK handles pooling
     - OpenAI: LangChain handles pooling
```

### 10.3 Cost Optimization

```
Expensive Operations:
  1. LLM calls (most expensive)
  2. Embeddings (per unique query)
  3. Web search (via LLM)

Cost Reduction:
  1. Use gpt-4o-mini (cheaper than gpt-4)
  2. Temperature = 0.0 (deterministic)
  3. Batch operations when possible
  4. Cache aggressively
  5. Monitor token usage
  6. Limit web search to educated fallback

Estimated Costs (per conversation):
  - 5 turns × 5 LLM calls = 25 LLM invocations
  - ~10,000 tokens per conversation
  - gpt-4o-mini: $0.15 per 1M input tokens
  - = ~$0.0015 per conversation
  - × 1000 users = $1.50/day
```

---

## 11. Deployment Architecture

### 11.1 Docker Compose Setup

```yaml
version: "3.9"

services:
  # Main API
  api:
    build: .                    # Dockerfile in repo
    ports:
      - "8000:8000"            # Expose port
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - REDIS_URL=redis://redis:6379
      - MONGO_URI=mongodb://mongodb:27017
    depends_on:
      - redis
      - mongodb
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # MongoDB
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password

volumes:
  redis_data:
  mongo_data:
```

### 11.2 Scaling Considerations

```
Vertical Scaling (More Power):
  ├─ Increase worker count (uvicorn -w N)
  ├─ Increase Redis memory
  ├─ Increase MongoDB RAM
  └─ Works for moderate load

Horizontal Scaling (More Servers):
  ├─ Run API on multiple servers (behind load balancer)
  ├─ Shared Redis (same instance)
  ├─ Shared MongoDB (same instance)
  └─ Requires:
      ├─ Redis cluster or AWS ElastiCache
      ├─ MongoDB sharding or Atlas
      ├─ Load balancer (nginx, AWS ALB)
      └─ Session affinity (optional)
```

---

## 12. Monitoring & Observability

### 12.1 Logging Strategy

```python
# Structured logging throughout:
logger.info("Query analyzed: type=%s, translated='%s'", query_type, translated)
logger.warning("Classification failed: %s, defaulting to curriculum_specific", exc)
logger.error("Retrieval failed: %s", exc)
logger.debug("Documents retrieved: %d, avg_score: %.2f", len(docs), avg_score)

# Key metrics logged:
#   - Query type (conversational/educational)
#   - Number of documents retrieved
#   - LLM call duration
#   - Validation results
#   - Cache hits/misses
#   - Error messages with context
```

### 12.2 Metrics to Track

```
Real-time Monitoring:
  - API response time (target: < 5 seconds)
  - Error rate (target: < 1%)
  - Cache hit rate (target: > 50%)
  - Active sessions (Redis keys)
  - MongoDB collection size

Business Metrics:
  - User engagement (new/returning)
  - Query types distribution
  - Agent usage (student/teacher/conversational)
  - Language distribution
  - Validation pass/fail rate

Performance Metrics:
  - Embedding latency
  - Retrieval latency
  - LLM inference latency
  - Translation latency
  - Total request latency
```

---

## Summary

This technical deep dive covers:

1. **Complete architecture**: All systems and their interactions
2. **Node-by-node breakdown**: What each part does
3. **Hybrid retrieval**: Dense + sparse search mechanics
4. **Memory management**: Token trimming, summaries, dual storage
5. **Agent orchestration**: Routing, ReAct loops, validation
6. **Error handling**: Fallbacks and recovery strategies
7. **Optimization**: Token efficiency, latency, cost reduction
8. **Deployment**: Docker setup and scaling strategies
9. **Monitoring**: Logging and metrics

**For New Developers**: Start with the quick reference, then this deep dive, then the code comments.

**For System Architects**: Use this to understand integration points and scaling strategies.

**For DevOps**: Focus on sections 11 (Deployment) and 12 (Monitoring).
