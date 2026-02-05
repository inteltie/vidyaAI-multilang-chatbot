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
    model_name: str = Field("gpt-4o-mini", description="LLM Model Name")
    web_search_model_name: str = Field("", description="Web Search Model Name")
    validator_model_name: str = Field("gpt-4o-mini", description="Validator Model Name")
    embedding_model: str = Field("text-embedding-3-large", description="Embedding Model Name")
    llm_temperature: float = Field(0.0, description="LLM Temperature")

    # -- Application Settings --
    # Memory
    max_tokens_default: int = Field(500, description="Default max tokens for LLM response")
    max_tokens_detailed: int = Field(1000, description="Max tokens for detailed responses")
    max_tokens_brief: int = Field(300, description="Max tokens for brief responses")
    
    # Specific Caps
    query_analysis_tokens: int = Field(100, description="Max tokens for query analysis")
    main_response_tokens: int = Field(2000, description="Max tokens for main agent response")
    validation_tokens: int = Field(300, description="Max tokens for response validation")
    memory_buffer_size: int = Field(10, description="Number of turns to keep in memory buffer")
    memory_token_limit: int = Field(2000, description="Max tokens for conversation history buffer")
    
    # Agents
    max_iterations: int = Field(5, description="Max ReAct agent iterations")
    web_search_enabled: bool = Field(True, description="Enable/disable web search tool")
    validation_mode: Literal["strict", "fast", "disabled"] = Field("fast", description="Groundedness validation mode")
    enable_query_caching: bool = Field(True, description="Enable caching for query analysis")
    cache_size: int = Field(1000, description="Size of query analysis cache")
    parallel_rag_fetch: bool = Field(True, description="Enable parallel RAG retrieval")
    
    # Retrieval
    retriever_top_k: int = Field(5, description="Number of documents to retrieve")
    retriever_score_threshold: float = Field(0.4, description="Minimum similarity score for retrieval")

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
            "mongo_db_name": os.getenv("MONGO_DB_NAME") or os.getenv("DB_NAME"),
            
            "model_name": os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
            "web_search_model_name": os.getenv("WEB_SEARCH_MODEL_NAME") or "gpt-4o-mini",
            "validator_model_name": os.getenv("VALIDATOR_MODEL_NAME") or "gpt-4o-mini",
            "embedding_model": os.getenv("EMBEDDING_MODEL") or "text-embedding-3-large",
            "llm_temperature": float(os.getenv("LLM_TEMPERATURE") or 0.0),
            
            "max_tokens_default": int(os.getenv("MAX_TOKENS_DEFAULT") or 1500),
            "max_tokens_detailed": int(os.getenv("MAX_TOKENS_DETAILED") or 3000),
            "max_tokens_brief": int(os.getenv("MAX_TOKENS_BRIEF") or 800),
            
            "query_analysis_tokens": int(os.getenv("QUERY_ANALYSIS_TOKENS") or 100),
            "main_response_tokens": int(os.getenv("MAIN_RESPONSE_TOKENS") or 2000),
            "validation_tokens": int(os.getenv("VALIDATION_TOKENS") or 300),
            "memory_buffer_size": int(os.getenv("MEMORY_BUFFER_SIZE") or 20),
            "memory_token_limit": int(os.getenv("MEMORY_TOKEN_LIMIT") or 2000),
            
            "max_iterations": int(os.getenv("MAX_ITERATIONS") or 5),
            "web_search_enabled": str_to_bool(os.getenv("WEB_SEARCH_ENABLED"), True),
            "validation_mode": (os.getenv("VALIDATION_MODE") or "fast").lower(),
            "enable_query_caching": str_to_bool(os.getenv("ENABLE_QUERY_CACHING", "True")),
            "cache_size": int(os.getenv("CACHE_SIZE") or 1000),
            "parallel_rag_fetch": str_to_bool(os.getenv("PARALLEL_RAG_FETCH"), True),
            
            "retriever_top_k": int(os.getenv("RETRIEVER_TOP_K") or 5),
            "retriever_score_threshold": float(os.getenv("RETRIEVER_SCORE_THRESHOLD") or 0.4),
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
