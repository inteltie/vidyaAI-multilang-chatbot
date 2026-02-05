import logging
import json
from typing import Any, Dict, Optional, List

from openai import AsyncOpenAI

from tools.base import Tool
from config import settings
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """
    Tool for searching the web using OpenAI's native web search capability.
    
    Uses the OpenAI Responses API with web_search_preview tool type.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information, facts, or topics. Uses OpenAI's native search which provides high-quality summaries with citations. Use this when you need up-to-date information."
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query to find information on the web",
                "required": True,
            },
        }
    
    async def execute(self, query: str, **kwargs) -> tuple[str, int, int]:
        """
        Execute web search using OpenAI's native capability.
        
        Args:
            query: Search query string
            
        Returns:
            String with rich search results summary and citations.
        """
        try:
            # Check cache
            cache_key = CacheService.generate_key("web_search", query)
            cached_result = await CacheService.get(cache_key)
            if cached_result:
                logger.info("Cache HIT for web search: %s", query)
                return cached_result, 0, 0
            
            logger.info("Executing web search for: %s", query[:100])
            
            # Use OpenAI's Responses API
            # Note: We use the 'web_search_preview' tool type which triggers the native search
            response = await self._client.responses.create(
                model=settings.web_search_model_name or "gpt-4o-mini", # Fallback to mini if not set
                input=f"Please search the web for: {query}",
                max_output_tokens=200,
                tools=[{
                    "type": "web_search_preview"
                }],
            )
            
            # Log token usage
            i_tokens, o_tokens = 0, 0
            if hasattr(response, 'usage') and response.usage:
                i_tokens = response.usage.input_tokens
                o_tokens = response.usage.output_tokens
                logger.info(
                    "[TOKEN_USAGE] WebSearchTool: input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                    i_tokens,
                    o_tokens,
                    response.usage.total_tokens,
                    settings.web_search_model_name or "gpt-4o-mini"
                )
            
            # Parse the response structure
            # The structure is typically: response.output which is a list.
            # We look for 'message' type output which contains the text and annotations.
            
            final_text = ""
            citations = []
            
            if hasattr(response, 'output'):
                for item in response.output:
                    # We are looking for the 'message' type which has the answer
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content'):
                            for content_part in item.content:
                                if hasattr(content_part, 'type') and content_part.type in ['text', 'output_text']:
                                    final_text += content_part.text
                                    
                                    # Extract citations from annotations
                                    if hasattr(content_part, 'annotations'):
                                        for annotation in content_part.annotations:
                                            if hasattr(annotation, 'type') and annotation.type == 'url_citation':
                                                citation_str = f"[{annotation.title}]({annotation.url})"
                                                if citation_str not in citations:
                                                    citations.append(citation_str)
            
            if not final_text:
                logger.warning("No text content found in web search response: %s", response)
                return "No information found for this query.", i_tokens, o_tokens
            
            # Append collected citations if they aren't already embedded nicely
            # (OpenAI usually embeds them as [1], [2] etc, but adding a sources list is helpful)
            result_str = final_text
            if citations:
                result_str += "\n\n**Sources:**\n" + "\n".join([f"- {c}" for c in citations])
            
            final_result = f"WEB_SEARCH_OBSERVATION for '{query}':\n{result_str}"
            
            # Cache result (TTL 24 hours)
            await CacheService.set(cache_key, final_result, ttl=86400)
            
            return final_result, i_tokens, o_tokens
            
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            return f"Web search failed: Could not retrieve results for '{query}'. Error: {str(exc)}", 0, 0


__all__ = ["WebSearchTool"]
