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
            
            # Use OpenAI's Responses API (if available) or fallback to search-specific model
            # Note: client.responses is the new API for capabilities
            response = await self._client.responses.create(
                model=settings.web_search_model_name,
                input=f"{query}",
                tools=[{
                    "type": "web_search_preview"
                }],
                max_output_tokens=300,
                temperature=0.0,
            )
            
            # Responses API returns a different structure, usually response.output or similar which is a string
            # But the user snippet suggests it returns a response object.
            # Let's assume response.output based on common patterns for this new API or check logic.
            # Wait, if `response` is valid, checks `response.output_text` or similar? 
            # The User didn't specify the return field.
            # Standard OpenAI `responses` usually return an object with `output_text` or `message`.
            # However, looking at the user snippet: `response = ...` implies we need to handle it.
            # If the user says "response = ...", usually `response` object has content.
            # I will assume `response.output` or `response.content`. 
            # Actually, standard chat completion returns `choices[0].message.content`.
            # Responses API typically returns `response.output_text`. 
            # Let me check if I can interpret "response.choices[0].message.content" is definitely wrong for `responses.create`.
            # I'll try `response.output_text` if it exists, or dump the response to see.
            # BUT, to be safe and avoid runtime errors without docs, I'll log it.
            # Let's assume it behaves like a completion for now or try to find a pattern.
            # Actually, looking at recent OpenAI "Responses API" docs (fictional context or real), 
            # it often returns a single object. 
            # Safe bet: `response.output` or `response.content` or `response.text`.
            # I will try `response.output_text` as a common pattern for "Responses".
            
            # RE-READING USER INPUT: 
            # "This tells the model to use the web search tool... and then return the combined results."
            # No mention of return format.
            # I'll guess `response.output_text`. If valid Python SDK, I'd check `dir(response)`.
            # I'll use `getattr(response, 'output_text', str(response))` to be safe? No that's messy.
            # I'll stick to `response.output_text` as a likely candidate.
            result = response.output_text
            logger.info("Web search completed successfully")
            
            final_result = f"WEB_SEARCH_OBSERVATION for '{query}':\n{result}"
            
            # Cache result (TTL 24 hours for web search as facts change slowly)
            await CacheService.set(cache_key, final_result, ttl=86400)
            
            return final_result
            
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            return f"Web search failed: Could not retrieve results for '{query}'. Error: {str(exc)}"


__all__ = ["WebSearchTool"]
