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
    
    # Educational subject keywords for fast-path heuristic
    _SUBJECT_KEYWORDS = {
        # Sciences
        "physics", "chemistry", "biology", "science", "botany", "zoology",
        "anatomy", "ecology", "genetics", "microbiology", "biochemistry",
        # Maths
        "mathematics", "maths", "math", "algebra", "geometry", "calculus",
        "trigonometry", "statistics", "probability", "arithmetic",
        # Social
        "history", "geography", "civics", "economics", "economy", "political",
        "sociology", "psychology", "philosophy", "anthropology",
        # Language / Lit
        "english", "hindi", "literature", "grammar", "essay", "poem", "poetry",
        "marathi", "sanskrit", "urdu",
        # CS / Tech
        "computer", "programming", "coding", "algorithm", "software",
        "hardware", "networking", "database", " python", "java",
        # Chemistry / Advanced Math
        "chemical kinetics", "chemical equations", "trigonometry", "stoichiometry",
        "calculus", "thermodynamics", "organic chemistry",
        # Generic educational
        "photosynthesis", "evolution", "atom", "molecule", "force", "gravity",
        "energy", "motion", "electricity", "magnetism", "thermodynamics",
        "chapter", "lesson", "topic", "concept", "theory", "theorem",
        "formula", "equation", "definition",
        # Japanese Subjects / Keywords
        "物理", "化学", "生物", "科学", "数学", "代数", "幾何", "歴史", 
        "地理", "経済", "文学", "英語", "日本語", "プログラミング", "光合成",
        "重力", "エネルギー",
    }

    def _check_heuristics(self, query: str) -> QueryClassification | None:
        """Check if query can be classified by simple heuristics."""
        query_lower = query.lower().strip()
        words = query_lower.split()

        # 1. Greeting / social check
        from services.utils import is_greeting
        if is_greeting(query):
            return QueryClassification(
                query_type="conversational",
                translated_query=query,
                confidence=1.0,
                reasoning="Matched social heuristic (greeting/ack).",
                subjects=["General"]
            )

        # 2. Subject-keyword fast path — catches 'i need ... for/about <subject>'
        #    and any query that contains an explicit educational subject noun.
        subject_hits = [k for k in self._SUBJECT_KEYWORDS if k in query_lower]
        if subject_hits:
            detected_subject = subject_hits[0].capitalize()
            return QueryClassification(
                query_type="curriculum_specific",
                translated_query=query,
                confidence=0.95,
                reasoning=f"Fast-path: detected subject keyword '{detected_subject}'.",
                subjects=[detected_subject]
            )

        # 3. Vague help requests — only fire when there is NO educational subject
        #    attached. Use exact-end check to avoid swallowing 'i need help with X'.
        help_patterns = [
            "i need help", "can you help me", "i need some help",
            "what can you do", "help me",
        ]
        # Only classify as conversational if the query ends with the help phrase
        # or the full query IS the help phrase (avoid false positives on 'i need help with economics').
        for pattern in help_patterns:
            if query_lower == pattern or query_lower.endswith(pattern):
                return QueryClassification(
                    query_type="conversational",
                    translated_query=query,
                    confidence=0.9,
                    reasoning="Matched meta-help request heuristic.",
                    subjects=["General"]
                )

        # 4. Simple acknowledgments
        if query_lower in {"ok", "okay", "alright", "sure", "fine", "k", "yep", "yes", "no"}:
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
        
        # 1. Check cache first (L1 memory, L2 Redis)
        if settings.enable_query_caching:
            history_text_for_hash = self._format_history(history, limit=2) # Only use 2 turns for cache key stability
            cache_key = hashlib.md5(f"{query}||{history_text_for_hash}".encode()).hexdigest()
            if cache_key in self._cache:
                logger.info("Found query classification in cache for: %s", query[:30])
                cached_result = self._cache[cache_key]
                return cached_result.model_copy(update={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning": f"Cached result. {cached_result.reasoning}"
                })
            try:
                from services.cache_service import CacheService
                redis_key = f"qclf:{cache_key}"
                cached = await CacheService.get(redis_key)
                if cached:
                    logger.info("Found query classification in Redis cache for: %s", query[:30])
                    cached_result = QueryClassification(**cached)
                    return cached_result.model_copy(update={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning": f"Cached result. {cached_result.reasoning}"
                    })
            except Exception as exc:
                logger.debug("Redis cache lookup failed for query classification: %s", exc)

        # 2. Try heuristics first (Zero LLM calls)
        heuristic_result = self._check_heuristics(query)
        if heuristic_result:
            logger.info("Heuristic classification: %s", heuristic_result.reasoning)
            return heuristic_result

        history_text = self._format_history(history, limit=5)
        
        prompt = f"""Analyze student query. 

Tasks:
1. **Standalone Query**: Reconstruct pronouns/follow-ups into complete English query (e.g., "Why?" -> "Why does photosynthesis happen?"). 
2. Translate to English (if needed).
3. Classify: "conversational" or "curriculum_specific":
   - conversational: greetings, meta-chat, vague help requests (e.g., "i need some help", "hi").
   - curriculum_specific: explicit educational topics OR asking for help ON an abstract academic subject (e.g., "help me with chemical kinetics", "explain gravity").
4. For "curriculum_specific", scan for: class_level, subjects, chapter, lecture_id.

History:
{history_text}

Query: {query}
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
                try:
                    from services.cache_service import CacheService
                    redis_key = f"qclf:{cache_key}"
                    await CacheService.set(redis_key, result.model_dump(), ttl=settings.query_cache_ttl)
                except Exception as exc:
                    logger.debug("Redis cache set failed for query classification: %s", exc)
            
            return result
            
        except Exception as exc:
            logger.warning("Analysis failed: %s, defaulting to curriculum_specific/en", exc)
            return QueryClassification(
                query_type="curriculum_specific",
                translated_query=query,
                confidence=0.0,
                reasoning=f"Fallback due to error: {exc}"
            )
