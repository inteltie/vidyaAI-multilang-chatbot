# VidyaAI-AGENT

An intelligent educational chatbot powered by LangGraph, designed to provide personalized learning experiences for both students and teachers. The system uses a multi-agent architecture with RAG (Retrieval-Augmented Generation) capabilities and supports multilingual interactions.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for students, teachers, and conversational interactions
- **Intelligent Query Routing**: Automatically routes queries to the appropriate agent based on user type and query classification
- **RAG-Powered Responses**: Hybrid search using Pinecone (dense + sparse vectors) for accurate information retrieval
- **Multilingual Support**: Automatic language detection and translation for seamless cross-language communication
- **Context-Aware**: Maintains conversation history and session context using Redis and MongoDB
- **Scalable Design**: Built with FastAPI and containerized with Docker for easy deployment

## 🏗️ Architecture

### LangGraph Workflow

The system uses a sophisticated LangGraph workflow with the following nodes:

1. **Load Memory**: Retrieves conversation history from Redis/MongoDB.
2. **Analyze Query**: Merged step that performs language detection, English translation, intent classification, and **proactive RAG fetch** (optimization for speed).
3. **Parse Session Context**: Extracts and validates session metadata (class level, subject, chapter, lecture_id) from current query and history.
4. **Agent Routing**: Routes based on `query_type` and `user_type`:
   - **Conversational Agent**: Handles small talk, greetings, and acknowledgments.
   - **Student Agent**: (Standard or Interactive) Provides fact-first synthesis or socratic step-by-step tutoring.
   - **Teacher Agent**: Provides scholarly analysis and session content reviews.
5. **Groundedness Check**: Validates educational responses against retrieved documents to prevent hallucinations.
6. **Translate Response**: Translates the answer back to the user's preferred language if it's not already English.
7. **Save Memory**: Persists messages to Redis and MongoDB in the background.

### Technology Stack

- **Framework**: FastAPI + LangGraph
- **LLM**: OpenAI GPT-4o-mini
- **Vector Database**: Pinecone (hybrid dense + sparse search)
- **Memory**: Redis (active sessions) + MongoDB (persistent storage)
- **Embeddings**: OpenAI text-embedding-3-large
- **Containerization**: Docker + Docker Compose

## 📋 Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- Pinecone API key
- Redis (included in Docker Compose)
- MongoDB (included in Docker Compose)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VidyaAI-AGENT
```

### 2. Set Up Environment Variables

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=vidyaai-agent-index
PINECONE_ENVIRONMENT=us-east-1

# Database Settings (for Docker)
MONGODB_URI=mongodb://mongodb:27017
DB_NAME=vidya_ai
REDIS_URL=redis://redis:6379

# LLM Settings
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0

# Retriever Settings
RETRIEVER_TOP_K=5
RETRIEVER_SCORE_THRESHOLD=0.4

# Memory Settings
MEMORY_BUFFER_SIZE=20

# Agent Settings
MAX_ITERATIONS=5
```

### 3. Run with Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app
```

The API will be available at `http://localhost:8000`

### 4. Run Locally (Alternative)

```bash
# Install dependencies using uv
pip install uv
uv sync

# Update database URLs in .env for local development
# MONGODB_URI=mongodb://localhost:27017
# REDIS_URL=redis://localhost:6379

# Start Redis and MongoDB separately
# Then run the application
python main.py
```

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "redis": true
}
```

### Chat Endpoint

```bash
POST /chat
```

**Request Body:**
```json
{
  "user_session_id": "unique-session-id",
  "user_id": "user-123",
  "user_type": "student",
  "query": "Explain photosynthesis",
  "language": "en",
  "agent_mode": "standard",
  "filters": {
    "subject_id": 101,
    "class_id": 10
  }
}
```

**Response:**
```json
{
  "user_session_id": "unique-session-id",
  "message": "Photosynthesis is the process by which plants...",
  "intent": "concept_explanation",
  "language": "en",
  "citations": [
    {
      "id": "doc-123",
      "score": 0.85,
      "subject_id": 101,
      "class_id": 10,
      "teacher_id": 55,
      "lecture_id": "456",
      "transcript_id": "789",
      "chunk_id": "012",
      "topics": "Biology, Plant Life"
    }
  ],
  "llm_calls": 2
}
```

**Request Fields:**
- `user_session_id` (required): Session identifier for conversation continuity
- `user_id` (required): Unique identifier for the user
- `user_type` (required): Either `"student"` or `"teacher"`
- `query` (required): User's message or question
- `language` (optional): ISO language code, defaults to `"en"`
- `agent_mode` (optional): `"standard"` (default) or `"interactive"`
- `filters` (optional): Metadata filters for vector search (e.g., `subject_id`, `class_id`, `lecture_id`)

**Response Fields:**
- `user_session_id`: Echo of the session identifier
- `message`: The chatbot's response text
- `intent`: Classified intent (e.g., `concept_explanation`, `homework_help`, `exam_prep`, `doubt_resolution`, `off_topic`)
- `language`: Language of the response
- `citations`: List of source documents used to generate the response
- `llm_calls`: Number of LLM API calls made to generate this response

## 🔧 Configuration

All configuration is managed through environment variables and centralized in `config.py`. Key settings include:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `LLM_TEMPERATURE` | `0.0` | Temperature for LLM responses |
| `RETRIEVER_TOP_K` | `5` | Number of documents to retrieve |
| `RETRIEVER_SCORE_THRESHOLD` | `0.4` | Minimum similarity score |
| `MEMORY_BUFFER_SIZE` | `20` | Conversation turns to keep in memory |
| `MAX_ITERATIONS` | `5` | Max iterations for ReAct agents |

## 📁 Project Structure

```
VidyaAI-AGENT/
├── agents/                 # Agent implementations
│   ├── student_agent.py   # Student-focused agent
│   ├── teacher_agent.py   # Teacher-focused agent
│   ├── conversational_agent.py
│   └── react_agent.py     # ReAct agent with tools
├── nodes/                  # LangGraph nodes
│   ├── analyze_query.py   # Query classification & translation
│   ├── load_memory.py     # Memory loading
│   ├── save_memory.py     # Memory persistence
│   ├── parse_session_context.py
│   ├── retrieve_documents.py
│   └── translate_response.py
├── services/              # Core services
│   ├── chat_memory.py    # Memory management (Redis + MongoDB)
│   ├── retriever.py      # Pinecone hybrid search
│   ├── translator.py     # LLM-based translation
│   ├── context_parser.py # Session context extraction
│   ├── query_classifier.py # Query analysis & routing
│   ├── response_validator.py # Groundedness checking
│   ├── citation_service.py # Citation extraction
│   └── utils.py          # Shared utility functions
├── tools/                 # Agent tools
│   ├── retrieval_tool.py # RAG retrieval tool
│   └── web_search_tool.py # Multi-source enrichment tool
├── models/               # Data models
│   ├── chat.py          # Chat models
│   └── domain.py        # Domain models
├── scripts/             # Utility scripts
├── config.py            # Centralized configuration
├── graph.py             # LangGraph workflow definition
├── state.py             # Agent state definition
├── main.py              # FastAPI application
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile           # Docker image definition
└── README.md           # This file
```

## 🧪 Development

### Running Tests

```bash
# Run tests (if available)
pytest
```

### Viewing Logs

```bash
# Docker logs
docker-compose logs -f app

# All services
docker-compose logs -f
```

### Accessing Databases

```bash
# MongoDB shell
docker-compose exec mongodb mongosh

# Redis CLI
docker-compose exec redis redis-cli
```

### Rebuilding After Code Changes

```bash
docker-compose up --build app
```

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart
```

### Port Conflicts

If ports 8000, 6379, or 27017 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

### Reset Everything

```bash
# Stop and remove all containers, networks, and volumes
docker-compose down -v

# Rebuild from scratch
docker-compose up --build -d
```

## 📚 Additional Documentation

- [Docker Setup Guide](DOCKER.md) - Detailed Docker configuration and troubleshooting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/)
- Vector search by [Pinecone](https://www.pinecone.io/)
