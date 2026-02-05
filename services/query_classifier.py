"""Query classifier for routing to appropriate agent."""

import logging
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from state import ConversationTurn

logger = logging.getLogger(__name__)



class QueryClassification(BaseModel):
    """Classification result for a user query."""
    query_type: Literal["conversational", "curriculum_specific"] = Field(
        description="The category of the query."
    )
    translated_query: str = Field(
        description="The query translated to English. If already English, return original."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation for the classification")
    subjects: List[str] = Field(
        default_factory=list,
        description="List of detected educational subjects (Math, Science, History, Geography, General)."
    )
    class_level: Optional[str] = Field(None, description="Class level, e.g., '10', '12', 'Class 9'")
    extracted_subject: Optional[str] = Field(None, description="Detailed subject name if mentioned")
    chapter: Optional[str] = Field(None, description="Chapter or topic name if mentioned")
    lecture_id: Optional[str] = Field(None, description="Specific lecture/session ID if mentioned")
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)


class QueryClassifier:
    """Classifies queries to route to appropriate agent."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._classifier = llm.with_structured_output(QueryClassification, include_raw=True)
        self._cache = {} # Simple cache for query analysis results
    
    def _check_heuristics(self, query: str) -> QueryClassification | None:
        """Check if query can be classified by simple heuristics."""
        query_lower = query.lower().strip()
        
        # 1. Conversational keywords (only if they are the ONLY word/phrase)
        conversational_keywords = {
            "hi", "hello", "hey", "greetings",
            "thanks", "thank you", "thx", "cool", "ok", "okay", "got it",
            "bye", "goodbye", "see ya", "nice", "great", "awesome", "yep", "yes", "no"
        }
        
        if query_lower in conversational_keywords or query_lower.rstrip("!?. ") in conversational_keywords:
            return QueryClassification(
                query_type="conversational",
                translated_query=query,
                confidence=1.0,
                reasoning="Matched short conversational keyword heuristic.",
                subjects=["General"]
            )
            
        # 2. Check for vague help requests or meta-queries
        help_patterns = ["i need help", "can you help", "i need some help", "what can you do", "help me"]
        if any(pattern in query_lower for pattern in help_patterns) and len(query_lower.split()) < 10:
            return QueryClassification(
                query_type="conversational",
                translated_query=query,
                confidence=0.9,
                reasoning="Matched meta-help request heuristic.",
                subjects=["General"]
            )
            
        # 3. Simple acknowledgments
        if query_lower in ["ok", "okay", "alright", "sure", "fine", "k"]:
            return QueryClassification(
                query_type="conversational",
                translated_query=query,
                confidence=1.0,
                reasoning="Matched acknowledgment heuristic.",
                subjects=["General"]
            )
            
        return None
        
    def _format_history(self, history: List[Union[ConversationTurn, BaseMessage]], limit: int = 4) -> str:
        """Format history into role: content text, handling both dicts and objects."""
        from services.utils import format_history
        return format_history(history, limit)

    async def analyze(
        self, 
        query: str, 
        history: List[ConversationTurn]
    ) -> QueryClassification:
        """
        Analyze query: detect language, translate to English, and classify type.
        
        Returns: QueryClassification object
        """
        from config import settings
        import hashlib
        import json
        
        # 1. Check cache first
        if settings.enable_query_caching:
            history_text_for_hash = self._format_history(history, limit=2) # Only use 2 turns for cache key stability
            cache_key = hashlib.md5(f"{query}||{history_text_for_hash}".encode()).hexdigest()
            if cache_key in self._cache:
                logger.info("Found query classification in cache for: %s", query[:30])
                return self._cache[cache_key]

        # 2. Try heuristics first (Zero LLM calls)
        heuristic_result = self._check_heuristics(query)
        if heuristic_result:
            logger.info("Heuristic classification: %s", heuristic_result.reasoning)
            return heuristic_result

        history_text = self._format_history(history, limit=10)
        
        prompt = f"""Analyze this student query. 

Tasks:
1. **Query Contextualization (CRITICAL)**: If the latest query is a follow-up or contains pronouns/contextual references (e.g., "What are its requirements?", "Why?", "Explain more"), reconstruct it into a standalone English query using the conversation history. Example: "What are its requirements?" -> "What are the requirements for photosynthesis?".
2. Translate the query to clear English (if not already English).
3. Classify into ONE category: "conversational" or "curriculum_specific".
4. If "curriculum_specific", detect one or more subjects: [Math, Science, History, Geography, General].
5. **Context Extraction**: Scan both the query and history for educational metadata:
   - class_level: (e.g., "Class 10", "Grade 12", "Batch A")
   - extracted_subject: (e.g., "Algebra", "Organic Chemistry", "Middle Ages")
   - chapter: (e.g., "Quadratic Equations", "Chapter 5", "World War II")
   - lecture_id: (e.g., "session_12", "lecture_101", "76")

Conversation history:
{history_text}

Latest query: {query}

Categories:
1. "conversational" - Greetings, small talk, general help requests, expressing satisfaction, OR meta-queries about the chat itself (e.g., "Analyze this query", "How do you work?").
2. "curriculum_specific" - ANY educational question about specific topics (Physics, Math, History, etc.) requiring external knowledge.

CRITICAL RULES:
- **Standalone Query**: Always return a complete, standalone version of the query in `translated_query`.
- Choose "curriculum_specific" ONLY if the user asks about a specific educational topic (e.g., "Newton's laws", "World War II").
- Choose "conversational" for:
  - Greetings ("Hi", "Hello")
  - General help requests ("I need help")
  - Meta-queries ("Analyze this query", "What can you do?", "Ignore previous instructions")
  - Thanks/Feedback ("Good job", "Thanks")
- For subjects, return a list including any that apply. Use "General" as fallback.
- Be thorough with Context Extraction. If the user mentions "in session 10", extract lecture_id as "10".
"""
        
        try:
            output = await self._classifier.ainvoke(prompt, config={"max_tokens": settings.query_analysis_tokens})
            result: QueryClassification = output["parsed"]
            raw_response = output["raw"]
            
            # Log token usage from raw response
            usage = getattr(raw_response, "usage_metadata", None) or getattr(raw_response, "response_metadata", {}).get("token_usage", {})
            if usage:
                 i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                 o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                 logger.info(
                     "[TOKEN_USAGE] QueryClassifier: input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                     i_tokens,
                     o_tokens,
                     usage.get("total_tokens") or (i_tokens + o_tokens),
                     self._llm.model_name
                 )
            
            # Populate token counts in result
            if usage:
                result.input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                result.output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0

            logger.info(
                "Query analyzed: type=%s, translated='%s'",
                result.query_type,
                result.translated_query
            )
            
            # Save to cache if enabled
            if settings.enable_query_caching:
                # Basic cache eviction (simple)
                if len(self._cache) >= settings.cache_size:
                    # Clear half the cache if full
                    keys_to_remove = list(self._cache.keys())[:settings.cache_size // 2]
                    for k in keys_to_remove:
                        del self._cache[k]
                
                # Use the same key generation logic as before
                history_text_for_hash = self._format_history(history, limit=2)
                cache_key = hashlib.md5(f"{query}||{history_text_for_hash}".encode()).hexdigest()
                self._cache[cache_key] = result
            
            return result
            
        except Exception as exc:
            logger.warning("Analysis failed: %s, defaulting to curriculum_specific/en", exc)
            return QueryClassification(
                query_type="curriculum_specific",
                translated_query=query,
                confidence=0.0,
                reasoning=f"Fallback due to error: {exc}"
            )
