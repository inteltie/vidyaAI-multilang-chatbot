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
    response_language: Optional[str] = Field(
        None, 
        description="The language code (e.g., 'hi', 'gu', 'en') if the user explicitly asks for a specific language in this query or if they switch languages."
    )


class QueryClassifier:
    """Classifies queries to route to appropriate agent."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._classifier = llm.with_structured_output(QueryClassification)
        self._cache = {} # Simple cache for query analysis results
    
    def _check_heuristics(self, query: str) -> str | None:
        """Check if query can be classified by simple heuristics."""
        query_lower = query.lower().strip()
        
        # Conversational keywords
        conversational_keywords = {
            "hi", "hello", "hey", "greetings",
            "thanks", "thank you", "thx", "cool", "ok", "okay", "got it",
            "bye", "goodbye", "see ya",
            "who are you", "what are you",
            "help", "i need help"
        }
        
        if query_lower in conversational_keywords:
            return "conversational"
            
        # Check for vague help requests that should be conversational
        if query_lower.startswith(("i need help", "can you help", "i need some help")):
            vague_terms = ["topic", "something", "study", "studies", "question", "doubt"]
            if any(term in query_lower for term in vague_terms) and len(query_lower.split()) < 10:
                return "conversational"
            
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
        # heuristic_type = self._check_heuristics(query)
        # if heuristic_type:
        #    # If heuristic works, we still need translation if not english...
        #    # For simplicity of merged step, let LLM handle everything unless cost is paramount.
        #    pass

        history_text = self._format_history(history, limit=4)
        
        prompt = f"""Analyze this student query. 

Tasks:
1. Translate the query to clear English (if not already English).
2. Classify into ONE category: "conversational" or "curriculum_specific".
3. If "curriculum_specific", detect one or more subjects: [Math, Science, History, Geography, General].
4. **Context Extraction**: Scan both the query and history for educational metadata:
   - class_level: (e.g., "Class 10", "Grade 12", "Batch A")
   - extracted_subject: (e.g., "Algebra", "Organic Chemistry", "Middle Ages")
   - chapter: (e.g., "Quadratic Equations", "Chapter 5", "World War II")
   - lecture_id: (e.g., "session_12", "lecture_101", "76")
5. **Language Switch Detection**: Check if the user specifically asks to change the response language (e.g., "Hindi mein samjhao", "Now answer in Gujarati", "Talk in English"). 
   - If detected, return the ISO 639-1 language code (hi, gu, mr, en, etc.) in response_language.

Conversation history:
{history_text}

Latest query: {query}

Categories:
1. "conversational" - Greetings, small talk, general help requests, or expressing satisfaction.
2. "curriculum_specific" - ANY educational question (DEFAULT - always use RAG).

CRITICAL RULES:
- If the query is educational in ANY way, choose "curriculum_specific".
- Use "conversational" for greetings, general help requests ("I need help"), and thanks.
- For subjects, return a list including any that apply. Use "General" as fallback.
- Be thorough with Context Extraction. If the user mentions "in session 10", extract lecture_id as "10".
"""
        
        try:
            result: QueryClassification = await self._classifier.ainvoke(prompt)
            
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
