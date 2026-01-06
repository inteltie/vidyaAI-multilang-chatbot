"""Redis-based caching service."""

import json
import logging
import hashlib
from typing import Any, Optional, Union

# Use redis.asyncio for async support
from redis.asyncio import Redis, RedisError

from config import settings

logger = logging.getLogger(__name__)

class CacheService:
    """
    Redis-based caching service for storing tool results and other expensive operations.
    Uses centralized configuration from config.settings.
    """
    
    _redis: Optional[Redis] = None
    
    @classmethod
    async def get_redis(cls) -> Redis:
        """Get or initialize Redis client."""
        if cls._redis is None:
            # Lazy initialization
            try:
                logger.info("Initializing Redis client.")
                cls._redis = Redis.from_url(
                    settings.redis_url, 
                    decode_responses=True, 
                    encoding="utf-8"
                )
            except Exception as e:
                logger.error("Failed to initialize Redis client: %s", e)
                raise
        return cls._redis

    @classmethod
    async def get(cls, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.
        Returns deserialized JSON object or None if miss/error.
        """
        try:
            redis = await cls.get_redis()
            data = await redis.get(key)
            if data:
                return json.loads(data)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning("Cache get failed for key %s: %s", key, e)
            return None
        except Exception as e:
            logger.warning("Unexpected error during cache get for key %s: %s", key, e)
            return None

    @classmethod
    async def set(cls, key: str, value: Any, ttl: int = 3600):
        """
        Set value in cache with TTL (default 1 hour).
        """
        try:
            redis = await cls.get_redis()
            # Handle Pydantic models or other non-serializable objects if necessary
            # For now assume standard JSON types
            serialized = json.dumps(value)
            await redis.setex(key, ttl, serialized)
        except (RedisError, TypeError) as e:
            logger.warning("Cache set failed for key %s: %s", key, e)
        except Exception as e:
            logger.warning("Unexpected error during cache set for key %s: %s", key, e)

    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """
        Generate a deterministic cache key from arguments.
        Format: prefix:sha256(args_representation)
        """
        # Create a consistent string representation
        # Sort kwargs to ensure deterministic output
        payload = f"{args}-{sorted(kwargs.items())}"
        hash_digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"

    @classmethod
    async def close(cls):
        """Close Redis connection."""
        if cls._redis:
            await cls._redis.close()
            cls._redis = None


__all__ = ["CacheService"]
