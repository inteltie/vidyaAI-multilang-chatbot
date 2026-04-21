"""LangGraph node: analyze_query."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict
from time import perf_counter

from services.query_classifier import QueryClassifier, QueryClassification
from services.language_detector import LanguageDetector
from state import AgentState

logger = logging.getLogger(__name__)


class AnalyzeQueryNode:
    """
    LangGraph node: analyze_query.
    
    Combines language detection, translation, and intent classification
    into a single step/LLM call.
    """

    def __init__(self, classifier: QueryClassifier, detector: LanguageDetector, retriever: RetrieverService | None = None) -> None:
        self._classifier = classifier
        self._detector = detector
        self._retriever = retriever

    @staticmethod
    def _detect_memory_intent(query: str) -> str | None:
        """Detect direct memory lookup questions that should avoid follow-up expansion."""
        q = (query or "").lower().strip()
        if not q:
            return None
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()

        name_patterns = (
            r"\bwhat is my name\b",
            r"\bwhats my name\b",
            r"\bwho am i\b",
            r"\bmy name\b",
        )
        class_patterns = (
            r"\bwhat is my class\b",
            r"\bwhats my class\b",
            r"\bwhich class am i in\b",
            r"\bwhat class am i in\b",
            r"\bi am in which class\b",
            r"\bmy class\b",
        )

        if any(re.search(pat, q) for pat in name_patterns):
            return "name"
        if any(re.search(pat, q) for pat in class_patterns):
            return "class_level"
        return None

    @staticmethod
    def _parse_language_preference_instruction(query: str) -> tuple[str | None, bool]:
        """Parse explicit user preference such as 'always reply in English'."""
        q = (query or "").lower().strip()
        if not q:
            return None, False
        normalized = re.sub(r"[^\w\s]", " ", q)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        explicit_markers = ("always", "from now on", "only", "reply in", "respond in", "speak in")
        if not any(marker in normalized for marker in explicit_markers):
            return None, False

        english_markers = ("english", "en")
        if any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in english_markers):
            return "en", True

        lang_map = {
            "hindi": "hi",
            "marathi": "mr",
            "japanese": "ja",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "norwegian": "no",
            "swedish": "sv",
        }
        for name, code in lang_map.items():
            if re.search(rf"\b{re.escape(name)}\b", normalized):
                return code, True
        return None, False

    @staticmethod
    def _extract_memory_language_preference(session_metadata: dict) -> str | None:
        structured = (session_metadata or {}).get("structured_summary") or {}
        if hasattr(structured, "model_dump"):
            structured = structured.model_dump()
        prefs = (structured or {}).get("user_preferences")
        if not prefs:
            return None
        pref_text = str(prefs).lower()
        if "english" in pref_text:
            return "en"
        return None

    async def __call__(self, state: AgentState) -> dict:
        start = perf_counter()
        updates = {
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        query = state["query"]
        history = state.get("conversation_history", [])
        user_session_id = state.get("user_session_id")
        clarification_attempts = int(state.get("context_gathering_attempts", 0) or 0)
        
        # Phase 5: Query Summarization for large inputs
        final_query = query
        try:
             if len(query) > 5000:
                 logger.info("Query exceeds 5000 chars. Summarizing for efficiency...")
                 summary_prompt = (
                     "Summarize the following user request into a concise search query, "
                     "preserving all technical constraints and educational context:\n\n"
                     f"{query}"
                 )
                 from config import settings
                 resp = await self._classifier._llm.ainvoke(summary_prompt, config={"max_tokens": settings.query_analysis_tokens})
                 
                 # Track tokens for summarization
                 usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
                 if usage:
                     updates["input_tokens"] += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                     updates["output_tokens"] += usage.get("output_tokens") or usage.get("completion_tokens") or 0

                 final_query = resp.content.strip()
                 logger.info("Summarized query: %s", final_query[:100])
             elif len(query) > 2000:
                 logger.info("Query between 2000-5000 chars. Using as-is (no summarization needed for model context window)")
                 final_query = query
        except Exception as e:
            logger.error("Query summarization failed: %s", e)

        # Phase 6: Sequential Analysis (Speculative RAG disabled for stability)
        import asyncio
        try:
            quality = self._classifier.assess_query_quality(final_query)
            updates["query_en"] = quality.normalized_query or final_query
            updates["clarification_message"] = None

            # Run social heuristics before unclear-query guardrails so greetings
            # (e.g., "hi") are not incorrectly routed to clarification.
            heuristic_result = self._classifier._check_heuristics(
                quality.normalized_query or final_query
            )
            logger.info(f"DEBUG: analyze_query heuristic_result: {heuristic_result}")

            if quality.is_unclear and clarification_attempts < 1 and not heuristic_result:
                clarification_text = quality.clarification_prompt or (
                    "Could you clarify what you mean so I can help you better?"
                )
                updates.update({
                    "query_type": "conversational",
                    "response": clarification_text,
                    "needs_context": True,
                    "clarification_message": clarification_text,
                    "context_gathering_attempts": clarification_attempts + 1,
                    "subjects": [],
                    "llm_calls": 0,
                    "detected_language": self._detector.detect_language(query),
                    "language": self._detector.detect_language(query),
                    "timings": {"analyze_query": perf_counter() - start},
                })
                logger.info("Unclear query detected (%s). Asking one clarification.", quality.reason)
                return updates

            # Heuristic: cheap follow-up resolution using cached context to avoid extra tokens
            followup_phrases = {
                "why", "how", "what", "explain", "explain more", "more",
                "elaborate", "tell me more", "again", "details", "detail",
                "give example", "examples", "example"
            }
            effective_query = quality.normalized_query or final_query
            query_norm = effective_query.lower().strip()
            memory_intent = self._detect_memory_intent(effective_query)
            if memory_intent:
                updates["memory_intent"] = memory_intent
            
            if heuristic_result:
                result = heuristic_result
                logger.info(f"DEBUG: analyze_query set result to heuristic: {result.query_type}")
            elif memory_intent:
                result = QueryClassification(
                    query_type="conversational",
                    translated_query=effective_query,
                    confidence=1.0,
                    reasoning=f"Matched direct memory intent ({memory_intent}).",
                    subjects=["General"],
                )
            else:
                is_followup = query_norm in followup_phrases
                cached_resolved = None
                if is_followup and user_session_id:
                    try:
                        from services.cache_service import CacheService
                        cached_resolved = await CacheService.get(f"resolved_query:{user_session_id}")
                    except Exception as e:
                        logger.debug("Resolved query cache lookup failed: %s", e)

                if cached_resolved and is_followup:
                    result = QueryClassification(
                        query_type="curriculum_specific",
                        translated_query=f"{cached_resolved}. Follow-up: {effective_query}",
                        confidence=0.8,
                        reasoning="Used cached resolved query for short follow-up.",
                        subjects=["General"]
                    )
                else:
                    # Classification task with 15s timeout
                    result = await asyncio.wait_for(self._classifier.analyze(effective_query, history), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("Query classification timed out after 15s")
            result = QueryClassification(
                query_type="curriculum_specific",
                translated_query=updates.get("query_en", final_query),
                confidence=0.0,
                reasoning="Fallback due to timeout",
                subjects=["General"]
            )
        except Exception as e:
            logger.error("Query classification failed: %s", e)
            result = QueryClassification(
                query_type="curriculum_specific",
                translated_query=updates.get("query_en", final_query),
                confidence=0.0,
                reasoning=f"Fallback due to error: {e}"
            )
        
        # Prepare updates
        updates.update({
            "query_en": result.translated_query,
            "query_type": result.query_type,
            "subjects": result.subjects,
            "is_acknowledgment": bool(
                result.query_type == "conversational"
                and self._classifier._normalize_chat_phrase(query) in self._classifier._ACK_TOKENS
            ),
            "llm_calls": 0 if (
                result.reasoning.startswith("Used cached resolved query") or
                result.reasoning.startswith("Cached result")
            ) else 1,
            "input_tokens": updates["input_tokens"] + result.input_tokens,
            "output_tokens": updates["output_tokens"] + result.output_tokens,
            "aux_input_tokens": updates.get("aux_input_tokens", 0) + result.input_tokens,
            "aux_output_tokens": updates.get("aux_output_tokens", 0) + result.output_tokens,
        })
        
        
        # Language detection and deterministic preference resolution
        detected_lang, detect_confidence = self._detector.detect_language_with_confidence(query)
        updates["detected_language"] = detected_lang
        updates["detected_language_confidence"] = detect_confidence

        session_metadata = state.get("session_metadata", {})
        prior_locked = bool(state.get("language_preference_locked", False))
        prior_preferred = state.get("preferred_response_language")
        memory_preferred = self._extract_memory_language_preference(session_metadata)
        parsed_preferred, parsed_locked = self._parse_language_preference_instruction(query)

        preferred_response_language = (
            parsed_preferred
            or prior_preferred
            or memory_preferred
            or "en"
        )
        language_preference_locked = parsed_locked or prior_locked or bool(memory_preferred)

        # Resolve target language with explicit precedence:
        # explicit/locked preference > memory preference > detected language.
        effective_language = preferred_response_language if language_preference_locked else detected_lang
        if not effective_language or str(effective_language).lower() == "null":
            effective_language = "en"
        updates["preferred_response_language"] = preferred_response_language
        updates["language_preference_locked"] = language_preference_locked
        updates["language"] = effective_language
        
        # Override state language if needed, or just keep as-is
        # Typically we want to know what language the current query is in regardless of request
        
        # Session metadata extraction
        if result.query_type != "conversational":
            current_meta = state.get("session_metadata", {})
            new_meta = current_meta.copy()
                
            if not new_meta.get("class_level") and result.class_level:
                new_meta["class_level"] = result.class_level
            if not new_meta.get("subject") and result.extracted_subject:
                new_meta["subject"] = result.extracted_subject
            if not new_meta.get("topics") and result.chapter:
                new_meta["topics"] = result.chapter
            if not new_meta.get("lecture_id") and result.lecture_id:
                new_meta["lecture_id"] = result.lecture_id
            
            updates["session_metadata"] = new_meta
            
            # -- SPECULATIVE RAG --
            # Trigger retrieval early if we have a translated query and subjects
            if self._retriever and result.translated_query:
                try:
                    from config import settings
                    if settings.parallel_rag_fetch:
                        logger.info("Executing speculative RAG for query: %s", result.translated_query[:50])
                        # We use the translated query and any extracted metadata for better retrieval
                        search_filters = state.get("request_filters", {}).copy()
                        if result.lecture_id:
                            search_filters["lecture_id"] = result.lecture_id
                        
                        # Trigger search (await here as it's fast, but could be backgrounded if needed)
                        docs = await self._retriever.retrieve(
                            query_en=result.translated_query,
                            filters=search_filters
                        )
                        updates["documents"] = docs
                        logger.info("Speculative RAG fetched %d documents", len(docs))
                except Exception as e:
                    logger.warning("Speculative RAG failed: %s", e)
            
        # If we had a heuristic match, mark that we skipped the LLM
        if result.confidence >= 1.0 and "Matched" in result.reasoning:
            updates["llm_calls"] = 0
        
        # Cache resolved query for cheap follow-up handling
        if user_session_id and result.translated_query:
            try:
                from services.cache_service import CacheService
                from config import settings
                await CacheService.set(
                    f"resolved_query:{user_session_id}",
                    result.translated_query,
                    ttl=settings.resolved_query_cache_ttl
                )
            except Exception as e:
                logger.debug("Resolved query cache set failed: %s", e)
        
        duration = perf_counter() - start
        updates["timings"] = {"analyze_query": duration}
        
        return updates
