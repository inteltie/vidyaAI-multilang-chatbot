"""FastAPI application entrypoint for the educational chatbot."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from contextlib import asynccontextmanager
from typing import Any, Dict

import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from graph import ChatbotGraphBuilder
from models import ChatRequest, ChatResponse, ErrorResponse, QueryIntent
from models import ChatSession
from nodes import (
    AnalyzeQueryNode,
    LoadMemoryNode,
    ParseSessionContextNode,
    SaveMemoryNode,
    TranslateResponseNode,
    GroundednessCheckNode,
    RetrieveDocumentsNode,
)
from services import (
    ContextParser,
    MemoryService,
    QueryClassifier,
    RetrieverService,
    Translator,
    CitationService,
    ResponseValidator,
    LanguageDetector,
)
from state import AgentState
from config import settings

logger = logging.getLogger(__name__)


class BackendApp:
    """Application factory that wires services, graph, and FastAPI routes."""

    def __init__(self) -> None:
        # Settings are loaded from environment variables
        self._settings = settings
        self._redis_client: aioredis.Redis | None = None
        self._graph = None
        self._language_detector = LanguageDetector()
        self._configure_logging()
        # Build app with lifespan
        self._app = self._build_app()
        self._configure_routes()

    @property
    def app(self) -> FastAPI:
        """Expose the underlying FastAPI app."""
        return self._app

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )

    def _build_app(self) -> FastAPI:
        """Build FastAPI app with lifespan context manager."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan context manager for startup and shutdown."""
            # Startup
            logger.info("Starting up application...")
            
            # Redis initialization
            for i in range(5):
                try:
                    self._redis_client = aioredis.from_url(
                        self._settings.redis_url,
                        decode_responses=True,
                    )
                    await self._redis_client.ping()
                    logger.info("Connected to Redis.")
                    break
                except Exception as e:
                    if i == 4:
                        logger.error(f"Failed to connect to Redis after 5 attempts: {e}")
                        raise
                    logger.warning(f"Redis connection attempt {i+1} failed, retrying in {2**i}s...")
                    await asyncio.sleep(2**i)
            
            # MongoDB initialization
            for i in range(5):
                try:
                    self._mongo_client = AsyncIOMotorClient(
                        self._settings.mongo_uri,
                        serverSelectionTimeoutMS=5000
                    )
                    await init_beanie(
                        database=self._mongo_client[self._settings.mongo_db_name],
                        document_models=[ChatSession]
                    )
                    logger.info("MongoDB connected and Beanie initialized.")
                    break
                except Exception as e:
                    if i == 4:
                        logger.error(f"Failed to initialize MongoDB after 5 attempts: {e}")
                        raise
                    logger.warning(f"MongoDB connection attempt {i+1} failed, retrying in {2**i}s...")
                    await asyncio.sleep(2**i)

            self._graph = self._build_graph()
            
            # -- WARMUP STEP --
            try:
                logger.info("Performing service warmup...")
                warmup_start = perf_counter()
                
                # 1. Warm up tokenizer (MemoryService)
                # We need access to memory_service. We can get it from the graph or by building it here.
                # To keep it simple, we'll build a temporary one or refactor _build_graph.
                # Let's just trigger a dummy call through the LLM used in the graph.
                llm = ChatOpenAI(
                    model=self._settings.model_name,
                    api_key=self._settings.openai_api_key,
                )
                from services import MemoryService, RetrieverService
                temp_mem = MemoryService(self._redis_client, llm)
                temp_retriever = RetrieverService(self._settings)
                
                await asyncio.gather(
                    temp_mem.warmup(),
                    # 2. Warm up Embeddings
                    asyncio.to_thread(temp_retriever._embeddings.embed_query, "Warmup"),
                    # 3. Warm up LLM
                    llm.ainvoke("hi")
                )
                
                logger.info("Service warmup completed in %.3fs", perf_counter() - warmup_start)
            except Exception as e:
                logger.warning(f"Service warmup encountered an issue: {e}")

            logger.info("LangGraph compiled and application started.")
            
            yield
            
            # Shutdown
            logger.info("Shutting down application...")
            if self._redis_client:
                await self._redis_client.aclose()
                logger.info("Redis client closed.")
            
            if hasattr(self, "_mongo_client") and self._mongo_client:
                self._mongo_client.close()
                logger.info("MongoDB client closed.")
        
        app = FastAPI(
            title="VidyaAI Educational Chatbot",
            version="0.1.0",
            lifespan=lifespan,
        )
        
        # Add CORS middleware to allow cross-origin requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins - restrict in production
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
            allow_headers=["*"],  # Allow all headers
        )
        
        return app

    def _build_graph(self):
        """Builds and compiles the LangGraph."""
        # Shared LLM client (deterministic responses with temperature 0.0)
        llm = ChatOpenAI(
            model=self._settings.model_name,
            api_key=self._settings.openai_api_key,
            temperature=0.0,
            max_tokens=self._settings.max_tokens_default,  # Default token limit
            max_retries=1,  # Reduced retries for stability
        )
        
        # LLM for validation (Fast and efficient for groundedness checks)
        llm_fast = ChatOpenAI(
            model=self._settings.validator_model_name,
            api_key=self._settings.openai_api_key,
            temperature=0.0,
            max_tokens=self._settings.max_tokens_default,
            max_retries=1,
        )

        # Services
        # self._redis_client is guaranteed to be not None here due to lifespan
        memory_service = MemoryService(self._redis_client, llm)
        
        # fastText-based language detector (no LLM calls)
        # language_detector = LanguageDetector()
        translator = Translator(llm)
        context_parser = ContextParser(llm)
        # intent_classifier removed as it was unused
        retriever_service = RetrieverService(self._settings)
        citation_service = CitationService()
        response_validator = ResponseValidator(llm_fast)

        
        # Agent services
        from agents import ConversationalAgent, StudentAgent, TeacherAgent, InteractiveStudentAgent
        
        query_classifier = QueryClassifier(llm)
        conversational_agent = ConversationalAgent(llm)
        
        # Student and Teacher agents with ReAct reasoning
        student_agent = StudentAgent(
            llm, 
            retriever_service, 
            max_iterations=settings.max_iterations,
            enable_web_search=settings.web_search_enabled
        )
        interactive_student_agent = InteractiveStudentAgent(
            llm, 
            retriever_service, 
            max_iterations=settings.max_iterations,
            enable_web_search=settings.web_search_enabled
        )
        teacher_agent = TeacherAgent(llm, retriever_service, max_iterations=settings.max_iterations)

        # Nodes
        load_memory_node = LoadMemoryNode(memory_service)
        
        # Agent nodes
        from nodes import (
            ConversationalAgentNode,
            StudentAgentNode,
            InteractiveStudentAgentNode,
            TeacherAgentNode,
        )
        
        analyze_query_node = AnalyzeQueryNode(query_classifier, self._language_detector, retriever_service)
        retrieve_documents_node = RetrieveDocumentsNode(retriever_service)
        conversational_agent_node = ConversationalAgentNode(conversational_agent)
        student_agent_node = StudentAgentNode(student_agent)
        interactive_student_agent_node = InteractiveStudentAgentNode(interactive_student_agent)
        teacher_agent_node = TeacherAgentNode(teacher_agent)
        groundedness_check_node = GroundednessCheckNode(response_validator)
        translate_response_node = TranslateResponseNode(translator)
        save_memory_node = SaveMemoryNode(memory_service)

        builder = ChatbotGraphBuilder(
            load_memory=load_memory_node,
            analyze_query=analyze_query_node,
            conversational_agent=conversational_agent_node,
            student_agent=student_agent_node,
            interactive_student_agent=interactive_student_agent_node,
            teacher_agent=teacher_agent_node,
            retrieve_documents=retrieve_documents_node,
            groundedness_check=groundedness_check_node,
            translate_response=translate_response_node,
            save_memory=save_memory_node,
        )
        return builder.compile()

    def _get_graph(self):
        if self._graph is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Graph not ready yet.",
            )
        return self._graph

    def _configure_routes(self) -> None:
        @self._app.get("/health")
        async def health() -> Dict[str, Any]:
            redis_ok = False
            try:
                if self._redis_client is not None:
                    await self._redis_client.ping()
                    redis_ok = True
            except Exception:
                redis_ok = False
            return {"status": "ok", "redis": redis_ok}

        @self._app.post(
            "/chat",
            response_model=ChatResponse,
            responses={500: {"model": ErrorResponse}},
        )
        async def chat_endpoint(request: ChatRequest, graph=Depends(self._get_graph))  -> ChatResponse:
            # 1. Enforce single graph execution per session (Stability Phase 1 - Distributed)
            session_id = request.user_session_id
            lock_key = f"lock:chat:{session_id}"
            lock_acquired = False
            
            try:
                if self._redis_client:
                    # Use SET with NX=True and EX=300 (5 minutes) for a distributed lock
                    is_locked = await self._redis_client.set(lock_key, "locked", nx=True, ex=300)
                    
                    if not is_locked:
                        logger.warning("Session %s is already being processed (Redis lock active). Rejecting duplicate.", session_id)
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail="This session is already being processed by another worker. Please wait.",
                        )
                    lock_acquired = True
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Redis distributed lock failed (Degraded Mode): {e}. Proceeding without lock.")
            
            try:
                # Initialize state
                state: AgentState = {
                    "query": request.query,
                    "user_id": request.user_id,
                    "user_session_id": request.user_session_id,
                    "user_type": request.user_type,
                    "language": self._language_detector.detect_language(request.query),
                    "agent_mode": request.agent_mode or "standard",
                    "student_grade": request.student_grade or "B",
                    "conversation_history": [],
                    "session_metadata": {},
                    "query_en": "",
                    "detected_language": "",  # To be filled by AnalyzeQueryNode if needed, or initialized here
                    "intent": QueryIntent.CONCEPT_EXPLANATION,
                    "is_context_reply": False,
                    "is_topic_shift": False,
                    "is_acknowledgment": False,
                    "query_type": "curriculum_specific",
                    "response": "",
                    "citations": [],
                    "llm_calls": 0,
                    "timings": {},
                    "is_session_restart": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                
                # Store UI filters separately (only for RAG retrieval)
                state["request_filters"] = request.filters.copy() if request.filters else {}
                if request.filters:
                    logger.info("Captured UI-provided filters: %s", request.filters)

                # Add timeout to prevent indefinite blocking
                graph_start = perf_counter()
                try:
                    final_state: AgentState = await asyncio.wait_for(
                        graph.ainvoke(state),
                        timeout=60.0  # 60 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Graph execution timed out after 60 seconds")
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail="Request timed out. Please try again.",
                    )
                
                graph_duration = perf_counter() - graph_start
                final_state["timings"]["total_graph"] = graph_duration
                logger.info("Total graph execution took %.3f seconds", graph_duration)
                
                # Global token usage log
                logger.info(
                    "[TOKEN_USAGE] TOTAL REQUEST CYCLE: input_tokens=%s, output_tokens=%s, total_tokens=%s",
                    final_state.get("input_tokens", 0),
                    final_state.get("output_tokens", 0),
                    final_state.get("input_tokens", 0) + final_state.get("output_tokens", 0)
                )

                message = final_state.get("response", "")
                
                # Map query_type to intent for the response
                query_type = final_state.get("query_type", "curriculum_specific")
                if query_type == "conversational":
                    intent = "conversational"
                else:
                    intent = final_state.get("intent", QueryIntent.CONCEPT_EXPLANATION).value
                
                language = final_state.get(
                    "final_language", final_state.get("detected_language", request.language)
                )
                citations = final_state.get("citations", []) or []
                timings = final_state.get("timings", {}) or {}
                llm_calls = int(final_state.get("llm_calls", 0) or 0)
                input_tokens = int(final_state.get("input_tokens", 0) or 0)
                output_tokens = int(final_state.get("output_tokens", 0) or 0)
                total_tokens = input_tokens + output_tokens

                # Include background summarization tokens since last response (if any)
                bg_input_tokens = 0
                bg_output_tokens = 0
                bg_total_tokens = 0
                try:
                    from services.cache_service import CacheService
                    bg = await CacheService.pop_hash(f"bg_tokens:{request.user_session_id}")
                    if bg:
                        bg_input_tokens = int(bg.get("input_tokens", 0))
                        bg_output_tokens = int(bg.get("output_tokens", 0))
                        bg_total_tokens = int(bg.get("total_tokens", 0))
                except Exception:
                    pass
                total_with_background = total_tokens + bg_total_tokens

                # Log per-step timings to the server log
                for step, duration in sorted(timings.items()):
                    logger.info("timing step=%s duration=%.3fs", step, duration)

                return ChatResponse(
                    user_session_id=request.user_session_id,
                    message=message,
                    intent=intent,
                    language=language,
                    citations=citations,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    background_input_tokens=bg_input_tokens,
                    background_output_tokens=bg_output_tokens,
                    background_total_tokens=bg_total_tokens,
                    total_tokens_with_background=total_with_background,
                )

            except HTTPException:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Graph execution failed: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An internal error occurred while processing the request.",
                ) from exc
            finally:
                # Always release the lock if it was acquired
                if lock_acquired:
                    try:
                        await self._redis_client.delete(lock_key)
                    except Exception:
                        pass
            


backend = BackendApp()
app = backend.app


if __name__ == "__main__":  # pragma: no cover - manual run helper
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
