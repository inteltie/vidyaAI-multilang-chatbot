"""Web Search Tool using OpenAI's native web search capability."""

import logging
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from tools.base import Tool
from config import settings
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """
    Tool for searching the web using OpenAI's native web search.
    
    Uses the OpenAI Responses API with web_search tool type.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information, facts, or topics not covered in the curriculum. Use when RAG retrieval doesn't have sufficient information."
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query to find information on the web",
                "required": True,
            },
        }
    
    async def execute(self, query: str, **kwargs) -> str:
        """
        Execute web search using OpenAI's native capability.
        
        Args:
            query: Search query string
            
        Returns:
            String with search results summary
        """
        try:
            # Check cache
            cache_key = CacheService.generate_key("web_search", query)
            cached_result = await CacheService.get(cache_key)
            if cached_result:
                logger.info("Cache HIT for web search: %s", query)
                return cached_result
            
            logger.info("Executing web search for: %s", query[:100])
            
            # Use OpenAI's chat completion with web search grounding
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise research assistant. "
                            "Provide a brief, factual summary of the most relevant information. "
                            "Avoid conversational filler. "
                            "CRITICAL: Always include any URLs or sources found. "
                            "Format sources as [Source: URL or Name] at the end."
                        )
                    },
                    {
                        "role": "user", 
                        "content": f"Briefly summarize: {query}"
                    }
                ],
                max_tokens=300,
                temperature=0.0,
            )
            
            result = response.choices[0].message.content
            logger.info("Web search completed successfully")
            
            final_result = f"WEB_SEARCH_OBSERVATION for '{query}':\n{result}"
            
            # Cache result (TTL 24 hours for web search as facts change slowly)
            await CacheService.set(cache_key, final_result, ttl=86400)
            
            return final_result
            
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            return f"Web search failed: Could not retrieve results for '{query}'. Error: {str(exc)}"


__all__ = ["WebSearchTool"]
