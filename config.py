"""Centralized configuration management."""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()


def str_to_bool(val: str | None, default: bool = True) -> bool:
    """Safely convert string to boolean."""
    if val is None or val == "":
        return default
    return val.lower() in ("true", "1", "t", "y", "yes")


def _strip_env_comment(value: str) -> str:
    """Strip inline comments from .env values (e.g. '512 # opts' -> '512')."""
    return (value or "").split("#", 1)[0].strip()


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    # -- External Services --
    openai_api_key: str = Field(..., description="OpenAI API Key")
    redis_url: str = Field(..., description="Redis Connection URL")
    
    # Pinecone
    pinecone_api_key: str = Field(..., description="Pinecone API Key")
    pinecone_index: str = Field(..., description="Pinecone Index Name")
    pinecone_env: str = Field("", description="Pinecone Environment (optional)")

    # MongoDB
    mongo_uri: str = Field(..., description="MongoDB Connection URI")
    mongo_db_name: str = Field(..., description="MongoDB Database Name")

    # -- Model Settings --
    model_name: str = Field("gpt-4o-mini", description="Main Agent LLM Model")
    web_search_model_name: str = Field("", description="Web Search Model Name")
    validator_model_name: str = Field("gpt-4.1-nano", description="Validator Model Name")
    analyzer_model_name: str = Field("gpt-4.1-nano", description="Query Analyzer Model Name")
    translator_model_name: str = Field("gpt-4.1-nano", description="Translator Model Name")
    summary_model_name: str = Field("gpt-4.1-nano", description="Background Summarizer Model Name")
    embedding_model: str = Field("text-embedding-3-large", description="Embedding Model Name")
    llm_temperature: float = Field(0.0, description="LLM Temperature")

    # -- Application Settings --
    # Memory
    max_tokens_default: int = Field(500, description="Default max tokens for LLM response")
    max_tokens_detailed: int = Field(1000, description="Max tokens for detailed responses")
    max_tokens_brief: int = Field(300, description="Max tokens for brief responses")
    
    # Understanding layer (runs before routing/generation)
    query_understanding_enabled: bool = Field(
        True, description="Run an LLM understanding pass before routing/generation"
    )
    query_understanding_timeout_s: float = Field(
        12.0, description="Max seconds for the LLM understanding pass"
    )
    query_understanding_min_confidence: float = Field(
        0.55, description="Min confidence before understanding overrides heuristics"
    )

    # Specific Caps
    query_analysis_tokens: int = Field(100, description="Max tokens for query analysis")
    main_response_tokens: int = Field(2000, description="Max tokens for main agent response")
    validation_tokens: int = Field(300, description="Max tokens for response validation")
    memory_buffer_size: int = Field(10, description="Number of turns to keep in memory buffer")
    memory_token_limit: int = Field(2000, description="Max tokens for conversation history buffer")
    memory_buffer_ttl: int = Field(7200, description="Redis buffer TTL (seconds)")
    summary_max_tokens: int = Field(400, description="Max tokens for conversation summary")
    max_chunk_tokens: int = Field(800, description="Max tokens per RAG chunk to prevent prompt bloating")
    
    # Agents
    max_iterations: int = Field(5, description="Max ReAct agent iterations")
    web_search_enabled: bool = Field(True, description="Enable/disable web search tool")
    validation_mode: Literal["strict", "fast", "disabled"] = Field("fast", description="Groundedness validation mode")
    enable_query_caching: bool = Field(True, description="Enable caching for query analysis")
    cache_size: int = Field(1000, description="Size of query analysis cache")
    query_cache_ttl: int = Field(3600, description="Query classification cache TTL (seconds)")
    resolved_query_cache_ttl: int = Field(14400, description="Resolved query cache TTL (seconds)")
    parallel_rag_fetch: bool = Field(True, description="Enable parallel RAG retrieval")
    web_rewrite_enabled: bool = Field(False, description="Enable query rewrite before web fallback search")
    web_search_fee_per_1k_calls_usd: float = Field(
        25.0, description="USD per 1000 billable web_search_preview tool calls (non-reasoning default)"
    )
    web_search_content_tokens_billed: bool | None = Field(
        None,
        description="Bill web-search content tokens; None=auto (reasoning models only)",
    )
    
    # Retrieval
    retriever_top_k: int = Field(5, description="Number of documents to retrieve")
    retriever_score_threshold: float = Field(0.4, description="Minimum similarity score for retrieval")
    rag_quality_high_score: float = Field(0.65, description="Score threshold for high-quality RAG")

    # Interactive image input
    max_images_per_request: int = Field(3, description="Max images per chat request")
    max_image_bytes: int = Field(5_000_000, description="Max decoded bytes per image")
    max_remembered_images: int = Field(9, description="Max uploaded image records per session")
    image_context_turn_window: int = Field(4, description="Turns for compact image context injection")
    image_gate_use_nano: bool = Field(True, description="Use nano LLM when image gate is ambiguous")
    vision_detail: Literal["low", "high", "auto"] = Field(
        "high", description="OpenAI vision detail for image analyzer (diagrams: high)"
    )
    image_generation_enabled: bool = Field(
        True, description="Allow interactive image generation requests"
    )
    gemini_api_key: str = Field(
        "", description="Google Gemini API key for diagram generation (GEMINI_API_KEY)"
    )
    google_image_model: str = Field(
        "gemini-3.1-flash-image",
        description="Google model for interactive image generation (Nano Banana 2)",
    )
    google_image_size: str = Field(
        "1K",
        description="Output resolution: 512, 1K, 2K, or 4K (Gemini 3.1 Flash Image)",
    )
    google_image_aspect_ratio: str = Field(
        "16:9",
        description="Aspect ratio for generated diagrams (e.g. 16:9, 4:3, 1:1)",
    )
    image_generation_fee_per_call_usd: float = Field(
        0.039, description="USD per successful Gemini-generated image (~$0.039/image)"
    )
    generated_image_dir: str = Field(
        "./data/generated",
        description="Directory for persisted generated diagram files",
    )
    public_base_url: str = Field(
        "http://localhost:8001",
        description="Public API base URL for generated image links",
    )
    max_remembered_generated_images: int = Field(
        20, description="Max generated image metadata records per session in Mongo"
    )
    mongo_session_load_timeout_s: float = Field(
        3.0,
        description="Max seconds to wait for per-request Mongo session load before Redis-only fallback",
    )
    graph_timeout_s: float = Field(
        120.0,
        description="Max seconds for full LangGraph /chat execution",
    )
    image_generation_timeout_s: float = Field(
        90.0,
        description="Max seconds to wait for Gemini image generation API",
    )
    query_classification_timeout_s: float = Field(
        15.0,
        description="Max seconds for LLM query classification",
    )
    uvicorn_workers: int = Field(
        2,
        description="Uvicorn worker processes (use 1-2 for local Docker)",
    )

    # Translation agent
    translation_use_primary_model: bool = Field(
        False, description="Use primary model instead of translator model"
    )
    translation_use_nano_judge: bool = Field(
        True, description="Use nano judge for ambiguous translation decisions"
    )
    translation_match_skip_confidence: float = Field(
        0.85, description="Skip translation when detected lang matches target"
    )
    translation_mismatch_translate_confidence: float = Field(
        0.70, description="Translate when mismatch confidence exceeds threshold"
    )
    language_response_min_confidence: float = Field(
        0.80,
        description=(
            "Minimum FastText confidence required before replying in a non-English "
            "detected language; otherwise respond in English."
        ),
    )

    class Config:
        """Pydantic configuration."""
        populate_by_name = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables with validation."""
        data = {
            # Map environment variables to fields
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "redis_url": os.getenv("REDIS_URL"),
            "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
            "pinecone_index": os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME"),
            "pinecone_env": os.getenv("PINECONE_ENV", ""),
            "mongo_uri": os.getenv("MONGO_URI") or os.getenv("MONGODB_URI"),
            "mongo_db_name": os.getenv("MONGO_DB") or os.getenv("MONGO_DB_NAME") or os.getenv("DB_NAME"),
            
            "model_name": os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
            "web_search_model_name": os.getenv("WEB_SEARCH_MODEL_NAME") or "gpt-4o-mini",
            "validator_model_name": os.getenv("VALIDATOR_MODEL_NAME") or "gpt-4.1-nano",
            "analyzer_model_name": os.getenv("ANALYZER_MODEL_NAME") or "gpt-4.1-nano",
            "translator_model_name": os.getenv("TRANSLATOR_MODEL_NAME") or "gpt-4.1-nano",
            "summary_model_name": os.getenv("SUMMARY_MODEL_NAME") or "gpt-4.1-nano",
            "embedding_model": os.getenv("EMBEDDING_MODEL") or "text-embedding-3-large",
            "llm_temperature": float(os.getenv("LLM_TEMPERATURE") or 0.0),
            
            "max_tokens_default": int(os.getenv("MAX_TOKENS_DEFAULT") or 1500),
            "max_tokens_detailed": int(os.getenv("MAX_TOKENS_DETAILED") or 3000),
            "max_tokens_brief": int(os.getenv("MAX_TOKENS_BRIEF") or 800),
            
            "query_understanding_enabled": str_to_bool(
                os.getenv("QUERY_UNDERSTANDING_ENABLED"), True
            ),
            "query_understanding_timeout_s": float(
                os.getenv("QUERY_UNDERSTANDING_TIMEOUT_S") or 12.0
            ),
            "query_understanding_min_confidence": float(
                os.getenv("QUERY_UNDERSTANDING_MIN_CONFIDENCE") or 0.55
            ),
            "query_analysis_tokens": int(os.getenv("QUERY_ANALYSIS_TOKENS") or 100),
            "main_response_tokens": int(os.getenv("MAIN_RESPONSE_TOKENS") or 2000),
            "validation_tokens": int(os.getenv("VALIDATION_TOKENS") or 300),
            "memory_buffer_size": int(os.getenv("MEMORY_BUFFER_SIZE") or 10),
            "memory_token_limit": int(os.getenv("MEMORY_TOKEN_LIMIT") or 2000),
            "memory_buffer_ttl": int(os.getenv("MEMORY_BUFFER_TTL") or 7200),
            "summary_max_tokens": int(os.getenv("SUMMARY_MAX_TOKENS") or 400),
            "max_chunk_tokens": int(os.getenv("MAX_CHUNK_TOKENS") or 800),
            
            "max_iterations": int(os.getenv("MAX_ITERATIONS") or 5),
            "web_search_enabled": str_to_bool(os.getenv("WEB_SEARCH_ENABLED"), True),
            "validation_mode": (os.getenv("VALIDATION_MODE") or "fast").lower(),
            "enable_query_caching": str_to_bool(os.getenv("ENABLE_QUERY_CACHING", "True")),
            "cache_size": int(os.getenv("CACHE_SIZE") or 1000),
            "query_cache_ttl": int(os.getenv("QUERY_CACHE_TTL") or 3600),
            "resolved_query_cache_ttl": int(os.getenv("RESOLVED_QUERY_CACHE_TTL") or 14400),
            "parallel_rag_fetch": str_to_bool(os.getenv("PARALLEL_RAG_FETCH"), True),
            "web_rewrite_enabled": str_to_bool(os.getenv("WEB_REWRITE_ENABLED"), False),
            "web_search_fee_per_1k_calls_usd": float(
                os.getenv("WEB_SEARCH_FEE_PER_1K_CALLS_USD") or 25.0
            ),
            "web_search_content_tokens_billed": (
                None
                if os.getenv("WEB_SEARCH_CONTENT_TOKENS_BILLED") is None
                else str_to_bool(os.getenv("WEB_SEARCH_CONTENT_TOKENS_BILLED"), False)
            ),
            
            "retriever_top_k": int(os.getenv("RETRIEVER_TOP_K") or 3),
            "retriever_score_threshold": float(os.getenv("RETRIEVER_SCORE_THRESHOLD") or 0.45),
            "rag_quality_high_score": float(os.getenv("RAG_QUALITY_HIGH_SCORE") or 0.65),
            "max_images_per_request": int(os.getenv("MAX_IMAGES_PER_REQUEST") or 3),
            "max_image_bytes": int(os.getenv("MAX_IMAGE_BYTES") or 5_000_000),
            "max_remembered_images": int(os.getenv("MAX_REMEMBERED_IMAGES") or 9),
            "image_context_turn_window": int(os.getenv("IMAGE_CONTEXT_TURN_WINDOW") or 4),
            "image_gate_use_nano": str_to_bool(os.getenv("IMAGE_GATE_USE_NANO"), True),
            "vision_detail": (
                vd
                if (vd := (os.getenv("VISION_DETAIL") or "high").lower())
                in ("low", "high", "auto")
                else "high"
            ),
            "image_generation_enabled": str_to_bool(os.getenv("IMAGE_GENERATION_ENABLED"), True),
            "gemini_api_key": (os.getenv("GEMINI_API_KEY") or "").strip(),
            "google_image_model": os.getenv("GOOGLE_IMAGE_MODEL") or "gemini-3.1-flash-image",
            "google_image_size": _strip_env_comment(os.getenv("GOOGLE_IMAGE_SIZE") or "1K"),
            "google_image_aspect_ratio": os.getenv("GOOGLE_IMAGE_ASPECT_RATIO") or "16:9",
            "image_generation_fee_per_call_usd": float(
                os.getenv("IMAGE_GENERATION_FEE_PER_CALL_USD") or 0.039
            ),
            "generated_image_dir": os.getenv("GENERATED_IMAGE_DIR") or "./data/generated",
            "public_base_url": os.getenv("PUBLIC_BASE_URL") or "http://localhost:8001",
            "max_remembered_generated_images": int(
                os.getenv("MAX_REMEMBERED_GENERATED_IMAGES") or 20
            ),
            "mongo_session_load_timeout_s": float(
                os.getenv("MONGO_SESSION_LOAD_TIMEOUT_S") or 3.0
            ),
            "graph_timeout_s": float(os.getenv("GRAPH_TIMEOUT_S") or 120.0),
            "image_generation_timeout_s": float(
                os.getenv("IMAGE_GENERATION_TIMEOUT_S") or 90.0
            ),
            "query_classification_timeout_s": float(
                os.getenv("QUERY_CLASSIFICATION_TIMEOUT_S") or 15.0
            ),
            "uvicorn_workers": int(os.getenv("UVICORN_WORKERS") or 2),
            "translation_use_primary_model": str_to_bool(
                os.getenv("TRANSLATION_USE_PRIMARY_MODEL"), False
            ),
            "translation_use_nano_judge": str_to_bool(
                os.getenv("TRANSLATION_USE_NANO_JUDGE"), True
            ),
            "translation_match_skip_confidence": float(
                os.getenv("TRANSLATION_MATCH_SKIP_CONFIDENCE") or 0.85
            ),
            "translation_mismatch_translate_confidence": float(
                os.getenv("TRANSLATION_MISMATCH_TRANSLATE_CONFIDENCE") or 0.70
            ),
            "language_response_min_confidence": float(
                os.getenv("LANGUAGE_RESPONSE_MIN_CONFIDENCE") or 0.80
            ),
        }
        
        # Filter out None values to allow Pydantic to raise validation errors for required fields
        # OR explicitly check for required ones here like before.
        # Pydantic validation is cleaner if we pass whatever we got.
        # However, our manual mapping means we are responsible for the 'None' being passed.
        
        try:
            return cls(**data)
        except ValidationError as e:
            # Re-raise with a clear message or let Pydantic handle it
            print("Configuration Error: Missing or invalid environment variables.")
            raise e

# Create global settings instance
try:
    settings = Settings.from_env()
    config = settings 
except ValidationError as e:
    # Print a very clear error and allow it to propagate if not in a build context
    print("FATAL: improperly configured environment variables.")
    print(e)
    # Re-raise so it's visible in docker logs
    raise e
except Exception as e:
    print(f"FATAL: unexpected error loading configuration: {e}")
    raise e
