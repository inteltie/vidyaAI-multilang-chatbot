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
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from graph import ChatbotGraphBuilder
from models import ChatRequest, ChatResponse, ErrorResponse, QueryIntent
from models.domain import (
    GeneratedImageMetadata,
    GeneratedImageOutput,
    ImageExplorationZoneOutput,
    MessageModalityTurn,
    SessionGeneratedImagesResponse,
    SessionMessageRecord,
    SessionMessagesResponse,
)
from services.message_modality import build_assistant_modality, build_user_modality
from models import ChatSession
from nodes import (
    AnalyzeQueryNode,
    LoadMemoryNode,
    ParseSessionContextNode,
    RouteDecisionNode,
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
    compute_request_cost,
    WebQueryRewriter,
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
        self._memory_service: MemoryService | None = None
        self._language_detector = LanguageDetector()
        self._configure_logging()
        # Build app with lifespan
        self._app = self._build_app()
        from services.generated_image_store import ensure_generated_image_dir

        generated_dir = ensure_generated_image_dir()
        self._app.mount(
            "/generated",
            StaticFiles(directory=str(generated_dir)),
            name="generated",
        )
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
            from services.generated_image_store import ensure_generated_image_dir

            ensure_generated_image_dir()
            
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
                llm_summary = ChatOpenAI(
                    model=self._settings.summary_model_name,
                    api_key=self._settings.openai_api_key,
                    temperature=0.0,
                    max_retries=1,
                )
                temp_mem = MemoryService(self._redis_client, llm, llm_summary)
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

        # Auxiliary LLMs for specific tasks mapped to cost-effective models like gpt-4.1-nano
        llm_analyzer = ChatOpenAI(
            model=self._settings.analyzer_model_name,
            api_key=self._settings.openai_api_key,
            temperature=0.0,
            max_retries=1,
        )

        llm_translator = ChatOpenAI(
            model=self._settings.translator_model_name,
            api_key=self._settings.openai_api_key,
            temperature=0.0,
            max_retries=1,
        )

        llm_summary = ChatOpenAI(
            model=self._settings.summary_model_name,
            api_key=self._settings.openai_api_key,
            temperature=0.0,
            max_retries=1,
        )

        # Services
        # self._redis_client is guaranteed to be not None here due to lifespan
        memory_service = MemoryService(self._redis_client, llm, llm_summary)
        self._memory_service = memory_service
        
        # fastText-based language detector (no LLM calls)
        # language_detector = LanguageDetector()
        translator = Translator(llm_translator)
        context_parser = ContextParser(llm)
        # intent_classifier removed as it was unused
        retriever_service = RetrieverService(self._settings)
        citation_service = CitationService()
        response_validator = ResponseValidator(llm_fast)

        
        # Agent services
        from agents import ConversationalAgent, StudentAgent, TeacherAgent, InteractiveStudentAgent
        
        query_classifier = QueryClassifier(llm_analyzer)
        web_query_rewriter = WebQueryRewriter(llm_analyzer)
        conversational_agent = ConversationalAgent(llm)
        
        # Student and Teacher agents with ReAct reasoning
        student_agent = StudentAgent(
            llm,
            retriever_service,
            max_iterations=settings.max_iterations,
            enable_web_search=settings.web_search_enabled,
            llm_fast=llm_analyzer,
        )
        from agents.translation_agent import TranslationAgent

        interactive_student_agent = InteractiveStudentAgent(
            llm,
            retriever_service,
            max_iterations=settings.max_iterations,
            enable_web_search=settings.web_search_enabled,
            llm_fast=llm_analyzer,
        )
        teacher_agent = TeacherAgent(llm, retriever_service, max_iterations=settings.max_iterations)

        # Nodes
        load_memory_node = LoadMemoryNode(memory_service)
        
        # Agent nodes
        from agents.image_generation_agent import ImageGenerationAgent
        from nodes import (
            ConversationalAgentNode,
            StudentAgentNode,
            InteractiveStudentAgentNode,
            TeacherAgentNode,
        )
        from nodes.image_generation_agent_node import ImageGenerationAgentNode
        
        understanding_service = None
        if getattr(settings, "query_understanding_enabled", True):
            from services.query_understanding import QueryUnderstandingService

            understanding_service = QueryUnderstandingService(
                llm_analyzer,
                timeout_s=getattr(settings, "query_understanding_timeout_s", 12.0),
            )
        from nodes import OrchestratorNode

        orchestrate_turn_node = OrchestratorNode(
            query_classifier,
            understanding=understanding_service,
        )
        analyze_query_node = AnalyzeQueryNode(
            query_classifier,
            self._language_detector,
            retriever_service,
            understanding=understanding_service,
        )
        route_decision_node = RouteDecisionNode()
        retrieve_documents_node = RetrieveDocumentsNode(retriever_service, web_query_rewriter)
        conversational_agent_node = ConversationalAgentNode(conversational_agent)
        student_agent_node = StudentAgentNode(student_agent)
        interactive_student_agent_node = InteractiveStudentAgentNode(interactive_student_agent)
        teacher_agent_node = TeacherAgentNode(teacher_agent)
        image_generation_agent_node = ImageGenerationAgentNode(
            ImageGenerationAgent(llm=llm)
        )
        groundedness_check_node = GroundednessCheckNode(response_validator)
        translation_agent = TranslationAgent(
            llm_translator if not settings.translation_use_primary_model else llm,
            llm_fast=llm_analyzer,
            detector=self._language_detector,
        )
        translate_response_node = TranslateResponseNode(translation_agent)
        save_memory_node = SaveMemoryNode(memory_service)

        builder = ChatbotGraphBuilder(
            load_memory=load_memory_node,
            orchestrate_turn=orchestrate_turn_node,
            analyze_query=analyze_query_node,
            conversational_agent=conversational_agent_node,
            student_agent=student_agent_node,
            interactive_student_agent=interactive_student_agent_node,
            teacher_agent=teacher_agent_node,
            retrieve_documents=retrieve_documents_node,
            route_decision=route_decision_node,
            groundedness_check=groundedness_check_node,
            translate_response=translate_response_node,
            save_memory=save_memory_node,
            image_generation_agent=image_generation_agent_node,
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
        @self._app.get(
            "/sessions/{user_session_id}/generated-images",
            response_model=SessionGeneratedImagesResponse,
        )
        async def session_generated_images(
            user_session_id: str,
            user_id: str,
        ) -> SessionGeneratedImagesResponse:
            if self._memory_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not ready yet.",
                )
            session = await ChatSession.find_one(
                ChatSession.session_id == user_session_id
            )
            if session and user_id and session.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="user_id does not match this session.",
                )
            raw = await self._memory_service.get_generated_images(user_session_id)
            items = [
                GeneratedImageMetadata(
                    image_id=str(r.get("image_id") or ""),
                    url=str(r.get("url") or ""),
                    mime_type=str(r.get("mime_type") or "image/png"),
                    prompt_used=str(r.get("prompt_used") or ""),
                    created_at=str(r.get("created_at") or ""),
                )
                for r in raw
            ]
            return SessionGeneratedImagesResponse(
                user_session_id=user_session_id,
                generated_images=items,
            )

        @self._app.get(
            "/sessions/{user_session_id}/messages",
            response_model=SessionMessagesResponse,
        )
        async def session_messages(
            user_session_id: str,
            user_id: str,
        ) -> SessionMessagesResponse:
            if self._memory_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not ready yet.",
                )
            session = await ChatSession.find_one(
                ChatSession.session_id == user_session_id
            )
            if not session:
                return SessionMessagesResponse(
                    user_session_id=user_session_id, messages=[]
                )
            if user_id and session.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="user_id does not match this session.",
                )
            records = []
            for m in session.messages or []:
                ts = m.timestamp.isoformat() if getattr(m, "timestamp", None) else ""
                records.append(
                    SessionMessageRecord(
                        role=m.role,
                        text=m.text,
                        timestamp=ts,
                        has_image_input=bool(getattr(m, "has_image_input", False)),
                        image_input_ids=list(getattr(m, "image_input_ids", None) or []),
                        has_generated_image=bool(getattr(m, "has_generated_image", False)),
                        generated_image_ids=list(
                            getattr(m, "generated_image_ids", None) or []
                        ),
                        generated_image_urls=list(
                            getattr(m, "generated_image_urls", None) or []
                        ),
                    )
                )
            return SessionMessagesResponse(
                user_session_id=user_session_id,
                messages=records,
            )

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
                    "preferred_response_language": "en",
                    "language_preference_locked": False,
                    "agent_mode": request.agent_mode or "standard",
                    "student_grade": request.student_grade or "B",
                    "conversation_history": [],
                    "session_metadata": {},
                    "query_en": "",
                    "detected_language": "",  # To be filled by AnalyzeQueryNode if needed, or initialized here
                    "detected_language_confidence": 0.0,
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
                    "token_usage_by_model": {},
                    "usage_by_component": {},
                    "web_search_calls": 0,
                    "images_analyzed": 0,
                    "image_modality": "none",
                    "orchestration_route": "retriever",
                    "route_reason": "",
                    "route_confidence": 0.0,
                    "route_fallback_reason": "",
                    "generated_images": [],
                    "image_generation_calls": 0,
                    "image_generation_backend": "",
                    "image_model_used": "",
                    "image_generation_status": "",
                    "image_generation_error_reason": "",
                    "embedding_tokens": 0,
                    "web_rewrite_enabled": False,
                    "web_query_original": "",
                    "web_query_used": "",
                    "rewrite_input_tokens": 0,
                    "rewrite_output_tokens": 0,
                    "rewrite_guard_reason": "",
                    "rewrite_overlap_ratio": 0.0,
                    "rewrite_used_original_due_to_guard": False,
                    "images": [img.model_dump() for img in (request.images or [])],
                    "has_images": bool(request.images),
                    "remembered_images": [],
                    "image_descriptions": [],
                    "image_context": "",
                    "effective_query": request.query,
                    "context_resolution_block": "",
                    "is_context_resolved_followup": False,
                    "wants_visual_output": False,
                    "wants_flowchart": False,
                    "response_strategy": "",
                    "preferred_response_format": "",
                    "query_understanding": {},
                    "turn_plan": {},
                    "image_inject_level": "none",
                    "image_inject_block": "",
                    "image_gate_reason": "",
                    "new_uploaded_image_records": [],
                    "new_generated_image_records": [],
                    "remembered_generated_images": [],
                    "interactive_turn_type": "opening",
                    "agent_findings": [],
                    "translation_action": "skip",
                    "translation_skip_reason": "",
                    "is_translated": False,
                }
                
                # Store UI filters separately (only for RAG retrieval)
                state["request_filters"] = request.filters.copy() if request.filters else {}
                if request.filters:
                    logger.info("Captured UI-provided filters: %s", request.filters)

                # Add timeout to prevent indefinite blocking
                graph_start = perf_counter()
                graph_timeout_s = float(self._settings.graph_timeout_s or 120.0)
                try:
                    final_state: AgentState = await asyncio.wait_for(
                        graph.ainvoke(state),
                        timeout=graph_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Graph execution timed out after %.0f seconds",
                        graph_timeout_s,
                    )
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
                elif query_type == "image_generation":
                    intent = "image_generation"
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
                token_usage_by_model = final_state.get("token_usage_by_model", {}) or {}
                usage_by_component = final_state.get("usage_by_component", {}) or {}
                web_search_calls = int(final_state.get("web_search_calls", 0) or 0)
                images_analyzed = int(final_state.get("images_analyzed", 0) or 0)
                image_generation_calls = int(final_state.get("image_generation_calls", 0) or 0)
                image_generation_backend = str(
                    final_state.get("image_generation_backend", "") or ""
                )
                image_model_used = str(final_state.get("image_model_used", "") or "")
                image_generation_status = str(final_state.get("image_generation_status", "") or "")
                image_generation_error_reason = str(
                    final_state.get("image_generation_error_reason", "") or ""
                )
                orchestration_route = str(final_state.get("orchestration_route", "") or "")
                route_reason = str(final_state.get("route_reason", "") or "")
                route_confidence = float(final_state.get("route_confidence", 0.0) or 0.0)
                route_fallback_reason = str(final_state.get("route_fallback_reason", "") or "")
                if (
                    not image_generation_status
                    and route_fallback_reason == "interactive_mode_required_for_image_generation"
                ):
                    image_generation_status = "interactive_recommended"
                raw_generated = final_state.get("generated_images") or []

                generated_images = []
                for img in raw_generated:
                    if not img.get("url"):
                        continue
                    zones_raw = img.get("exploration_zones") or []
                    zones = []
                    for z in zones_raw:
                        if not isinstance(z, dict):
                            continue
                        label = str(z.get("label") or "").strip()
                        if not label:
                            continue
                        zones.append(
                            ImageExplorationZoneOutput(
                                zone=int(z.get("zone") or len(zones) + 1),
                                label=label,
                                region_hint=str(z.get("region_hint") or ""),
                                curiosity_question=str(z.get("curiosity_question") or ""),
                            )
                        )
                    generated_images.append(
                        GeneratedImageOutput(
                            image_id=str(img.get("image_id") or ""),
                            url=str(img.get("url") or ""),
                            mime_type=img.get("mime_type", "image/png"),
                            prompt_used=img.get("prompt_used", ""),
                            created_at=str(img.get("created_at") or ""),
                            interactive_title=str(img.get("interactive_title") or ""),
                            exploration_zones=zones,
                        )
                    )
                embedding_tokens = int(final_state.get("embedding_tokens", 0) or 0)
                web_rewrite_enabled = bool(final_state.get("web_rewrite_enabled", False))
                web_query_original = str(final_state.get("web_query_original", "") or "")
                web_query_used = str(final_state.get("web_query_used", "") or "")
                rewrite_input_tokens = int(final_state.get("rewrite_input_tokens", 0) or 0)
                rewrite_output_tokens = int(final_state.get("rewrite_output_tokens", 0) or 0)
                rewrite_guard_reason = str(final_state.get("rewrite_guard_reason", "") or "")
                rewrite_overlap_ratio = float(final_state.get("rewrite_overlap_ratio", 0.0) or 0.0)
                rewrite_used_original_due_to_guard = bool(
                    final_state.get("rewrite_used_original_due_to_guard", False)
                )

                # Model-specific tracking
                primary_input_tokens = int(final_state.get("primary_input_tokens", 0) or 0)
                primary_output_tokens = int(final_state.get("primary_output_tokens", 0) or 0)
                aux_input_tokens = int(final_state.get("aux_input_tokens", 0) or 0)
                aux_output_tokens = int(final_state.get("aux_output_tokens", 0) or 0)

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
                cost_details = compute_request_cost(
                    usage_by_component=usage_by_component,
                    web_search_calls=web_search_calls,
                    image_generation_calls=image_generation_calls,
                    embedding_tokens=embedding_tokens,
                    embedding_model_name=self._settings.embedding_model,
                    background_input_tokens=bg_input_tokens,
                    background_output_tokens=bg_output_tokens,
                    background_model_name=self._settings.summary_model_name,
                    web_search_model_name=self._settings.web_search_model_name,
                    token_usage_by_model=token_usage_by_model,
                )
                logger.info(
                    "[COST] session=%s agent_mode=%s llm=$%.6f web=$%.6f vision=$%.6f "
                    "gen=$%.6f bg=$%.6f embed=$%.6f total=$%.6f web_calls=%d images=%d "
                    "gen_calls=%d gen_backend=%s gen_model=%s",
                    request.user_session_id,
                    request.agent_mode or "standard",
                    cost_details["cost_usd_llm"],
                    cost_details["cost_usd_web_search"],
                    cost_details["cost_usd_vision"],
                    cost_details["cost_usd_image_generation"],
                    cost_details["cost_usd_background"],
                    cost_details["cost_usd_embedding"],
                    cost_details["cost_usd_total"],
                    cost_details["web_search_calls"],
                    cost_details["images_analyzed"],
                    cost_details["image_generation_calls"],
                    image_generation_backend,
                    image_model_used,
                )
                logger.info(
                    "[ROUTE] route=%s confidence=%.2f reason=%s fallback=%s",
                    orchestration_route,
                    route_confidence,
                    route_reason[:180],
                    route_fallback_reason,
                )
                logger.info(
                    "[WEB_REWRITE] enabled=%s original=%r used=%r rewrite_input_tokens=%s rewrite_output_tokens=%s guard_reason=%s overlap=%.4f guard_fallback=%s",
                    web_rewrite_enabled,
                    web_query_original[:200],
                    web_query_used[:200],
                    rewrite_input_tokens,
                    rewrite_output_tokens,
                    rewrite_guard_reason,
                    rewrite_overlap_ratio,
                    rewrite_used_original_due_to_guard,
                )

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
                    primary_input_tokens=primary_input_tokens,
                    primary_output_tokens=primary_output_tokens,
                    aux_input_tokens=aux_input_tokens,
                    aux_output_tokens=aux_output_tokens,
                    embedding_tokens=embedding_tokens,
                    token_usage_by_model=token_usage_by_model,
                    cost_by_model_usd=cost_details["cost_by_model_usd"],
                    cost_usd_llm=cost_details["cost_usd_llm"],
                    cost_usd_web_search=cost_details["cost_usd_web_search"],
                    cost_usd_vision=cost_details["cost_usd_vision"],
                    cost_usd_background=cost_details["cost_usd_background"],
                    cost_usd_embedding=cost_details["cost_usd_embedding"],
                    cost_usd_total=cost_details["cost_usd_total"],
                    cost_by_component_usd=cost_details["cost_by_component_usd"],
                    web_search_calls=cost_details["web_search_calls"],
                    images_analyzed=cost_details["images_analyzed"],
                    generated_images=generated_images,
                    image_generation_calls=cost_details["image_generation_calls"],
                    cost_usd_image_generation=cost_details["cost_usd_image_generation"],
                    image_generation_backend=image_generation_backend,
                    image_model_used=image_model_used,
                    image_generation_status=image_generation_status,
                    image_generation_error_reason=image_generation_error_reason,
                    orchestration_route=orchestration_route,
                    route_reason=route_reason,
                    route_confidence=route_confidence,
                    route_fallback_reason=route_fallback_reason,
                    web_rewrite_enabled=web_rewrite_enabled,
                    web_query_original=web_query_original,
                    web_query_used=web_query_used,
                    rewrite_input_tokens=rewrite_input_tokens,
                    rewrite_output_tokens=rewrite_output_tokens,
                    rewrite_guard_reason=rewrite_guard_reason,
                    rewrite_overlap_ratio=rewrite_overlap_ratio,
                    rewrite_used_original_due_to_guard=rewrite_used_original_due_to_guard,
                    user_message_had_image_input=build_user_modality(
                        {**final_state, "images": request.images}
                    )["has_image_input"],
                    assistant_message_has_generated_image=bool(generated_images),
                    message_modality_timeline=[
                        MessageModalityTurn(**row)
                        for row in (final_state.get("message_modality_timeline") or [])
                    ],
                    memory_degraded=bool(final_state.get("memory_degraded")),
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
