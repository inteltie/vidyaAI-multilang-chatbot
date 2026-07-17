"""LangGraph node: analyze_query."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict
from time import perf_counter

from config import settings
from services.query_classifier import (
    QueryClassifier,
    QueryClassification,
    normalize_classification_query_type,
)
from services.language_detector import LanguageDetector
from services.agent_findings import publish
from services.prompt_policy import detect_abuse
from state import AgentState

logger = logging.getLogger(__name__)


class AnalyzeQueryNode:
    """
    LangGraph node: analyze_query.
    
    Combines language detection, translation, and intent classification
    into a single step/LLM call.
    """

    def __init__(self, classifier: QueryClassifier, detector: LanguageDetector, retriever: RetrieverService | None = None, understanding=None) -> None:
        self._classifier = classifier
        self._detector = detector
        self._retriever = retriever
        # ``understanding`` is retained for constructor compatibility; the LLM
        # understanding/planning now runs in OrchestratorNode (orchestrate_turn).
        self._understanding = understanding

    # Pure greetings that should NEVER be reclassified as interactive follow-ups,
    # even mid-Socratic flow. Acknowledgments like "yes/ok/idk/hmm" are NOT here
    # so they continue to flow into the interactive agent.
    _PURE_GREETING_TOKENS = {
        "hi", "helo", "hey", "hello", "namaste", "नमस्ते", "हेलो",
        "bye", "goodbye", "godbye",
    }
    _CLOSURE_TOKENS = {
        "no further questions",
        "i have no further questions",
        "no more questions",
        "i have no more questions",
        "no questions",
        "thats all",
        "that's all",
        "that is all",
        "nothing else",
        "all good now",
        "all clear now",
        "lets stop here",
        "let's stop here",
        "we can stop here",
    }

    @classmethod
    def _is_closure_utterance(cls, query: str) -> bool:
        q_norm = re.sub(r"[^\w\s']", " ", (query or "").lower()).strip()
        q_norm = re.sub(r"\s+", " ", q_norm)
        return any(token in q_norm for token in cls._CLOSURE_TOKENS)

    @classmethod
    def _is_interactive_followup(cls, state: dict, query: str) -> bool:
        """
        True when the user is mid-Socratic-loop in interactive mode and their
        current message is a short reply (acknowledgment / confusion / follow-up)
        to a question Vidya just posed.

        Used to override the conversational/acknowledgment heuristic so short
        replies do NOT leak out of the interactive agent.
        """
        if (state or {}).get("agent_mode") != "interactive":
            return False

        history = (state or {}).get("conversation_history") or []
        if not history:
            return False

        last_assistant_content: str | None = None
        for msg in reversed(history):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content") or msg.get("text") or ""
            else:
                role = getattr(msg, "type", None) or getattr(msg, "role", None)
                content = getattr(msg, "content", "") or ""
                # langchain BaseMessage uses type="ai" for assistant
                if role == "ai":
                    role = "assistant"
                elif role == "human":
                    role = "user"
            if role == "assistant" and content:
                last_assistant_content = str(content)
                break

        if not last_assistant_content:
            return False

        # Strip trailing emojis/whitespace and check for a question mark suffix.
        stripped = re.sub(r"[\s\W_]+$", "", last_assistant_content)
        # Re-add a trailing '?' check on the original (since '?' is in \W and got stripped);
        # use the original tail to detect whether the original ended with '?'.
        tail = re.sub(r"[\s\u200b]+$", "", last_assistant_content)
        ends_with_question = tail.endswith("?") or "?" in last_assistant_content[-3:]
        if not ends_with_question:
            return False

        # Skip pure greetings — student saying "hi" mid-flow is still a greeting.
        q_norm = re.sub(r"[^\w\s]", " ", (query or "").lower()).strip()
        q_norm = re.sub(r"\s+", " ", q_norm)
        q_collapsed = re.sub(r"(.)\1+", r"\1", q_norm)
        words = q_collapsed.split()
        if words and all(w in cls._PURE_GREETING_TOKENS for w in words):
            return False
        if cls._is_closure_utterance(query):
            return False

        return True

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
        )
        class_patterns = (
            r"\bwhat is my class\b",
            r"\bwhats my class\b",
            r"\bwhich class am i in\b",
            r"\bwhat class am i in\b",
            r"\bi am in which class\b",
        )

        if any(re.search(pat, q) for pat in name_patterns):
            return "name"
        if any(re.search(pat, q) for pat in class_patterns):
            return "class_level"
        return None

    @staticmethod
    def _extract_intro_entities(query: str) -> dict[str, str]:
        """Extract user-provided name/class from a self-introduction message."""
        text = (query or "").strip()
        if not text:
            return {}
        normalized = re.sub(r"\s+", " ", text)
        lowered = normalized.lower()

        entities: dict[str, str] = {}
        name_match = re.search(
            r"\b(?:my name is|i am|i'm)\s+([a-zA-Z][a-zA-Z]{1,29})\b",
            lowered,
        )
        if name_match:
            candidate = name_match.group(1).strip()
            if candidate not in {"in", "from", "class"}:
                entities["name"] = candidate.title()

        class_match = re.search(
            r"\bclass\s*(\d{1,2})(?:\s*(?:th|st|nd|rd))?\b",
            lowered,
        )
        if class_match:
            entities["class_level"] = f"Class {class_match.group(1)}"

        return entities

    @staticmethod
    def _build_intro_acknowledgement(intro_entities: dict[str, str]) -> str | None:
        """Create deterministic acknowledgement for user-provided profile details."""
        name = intro_entities.get("name")
        class_level = intro_entities.get("class_level")
        if name and class_level:
            return f"Thanks, {name}. Noted that you are in {class_level}. Please share your specific doubt and I will help."
        if name:
            return f"Thanks, {name}. I have noted your name. Please share your specific doubt and I will help."
        if class_level:
            return f"Noted that you are in {class_level}. Please share your specific doubt and I will help."
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
        if "preferred_response_language:" in pref_text:
            match = re.search(r"preferred_response_language:\s*([a-z]{2})", pref_text)
            if match:
                return match.group(1)
        lang_map = {
            "english": "en",
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
            if name in pref_text:
                return code
        return None

    async def __call__(self, state: AgentState) -> dict:
        start = perf_counter()
        updates = {
            "input_tokens": 0,
            "output_tokens": 0,
            "token_usage_by_model": {},
        }

        query = state.get("query", "") or ""
        has_images = bool(state.get("has_images") or (state.get("images") or []))
        history = state.get("conversation_history", [])
        user_session_id = state.get("user_session_id")
        clarification_attempts = int(state.get("context_gathering_attempts", 0) or 0)

        # The Turn Orchestrator (orchestrate_turn) already produced an authoritative,
        # context-aware plan. This node performs deterministic ENRICHMENT only:
        # language resolution, memory/intro handling, interactive-followup safety,
        # session-metadata extraction, caching and findings. It does NOT re-classify
        # intent — query_type/query_en/subjects come from the plan unless a hard
        # deterministic rule must override them.
        plan = state.get("turn_plan") or {}

        # Normalize runner-injected context wrappers into a stable input.
        if "\nCurrent question:" in query and query.strip().lower().startswith("context:"):
            try:
                tail = query.split("\nCurrent question:", 1)[1].strip()
                if tail:
                    query = tail
            except Exception:
                pass

        final_query = query

        # Baseline carried from the orchestrator plan (already in state).
        plan_query_type = str(
            state.get("query_type") or plan.get("query_type") or "curriculum_specific"
        )
        classification_query_type = normalize_classification_query_type(
            plan_query_type,
            has_images=has_images,
        )
        plan_query_en = str(
            state.get("query_en") or plan.get("standalone_query_en") or final_query
        )
        plan_subjects = list(state.get("subjects") or plan.get("subjects") or ["General"])
        plan_confidence = float(plan.get("confidence") or 0.0)
        wants_visual_eff = bool(
            state.get("wants_visual_output") or plan.get("wants_new_image")
        )
        is_time_sensitive_query = bool(
            state.get("is_time_sensitive_query") or plan.get("is_time_sensitive")
        )
        is_current_affairs_query = bool(state.get("is_current_affairs_query"))

        quality = self._classifier.assess_query_quality(plan_query_en or final_query)
        normalized_query = quality.normalized_query or plan_query_en or final_query
        heuristic_result = self._classifier._check_heuristics(normalized_query)
        interactive_followup = self._is_interactive_followup(state, final_query)
        if interactive_followup:
            logger.info(
                "Interactive follow-up detected; deterministic override keeps it in the educational flow."
            )

        updates["query_en"] = normalized_query
        updates["clarification_message"] = None

        # --- Deterministic unclear-query guard (gibberish/empty input) ---
        if (
            quality.is_unclear
            and clarification_attempts < 1
            and not heuristic_result
            and not interactive_followup
            and not has_images
            and not wants_visual_eff
            and plan_query_type != "image_generation"
        ):
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
                "is_time_sensitive_query": is_time_sensitive_query,
                "is_current_affairs_query": is_current_affairs_query,
            })
            logger.info("Unclear query detected (%s). Asking one clarification.", quality.reason)
            return updates

        # --- Deterministic abuse / scope gate ---
        # Short-circuit abusive, out-of-scope, or identity/tech-disclosure turns
        # with a friendly refuse+redirect before any retrieval/agent work. Skip
        # for interactive follow-ups and image turns to avoid mis-firing on short
        # Socratic replies or vision inputs.
        if not interactive_followup and not has_images:
            abuse_category, abuse_message = detect_abuse(final_query)
            if abuse_category and abuse_message:
                detected_lang = self._detector.detect_language(query)
                updates.update({
                    "query_type": "conversational",
                    "response": abuse_message,
                    "subjects": [],
                    "llm_calls": 0,
                    "detected_language": detected_lang,
                    "language": detected_lang,
                    "is_time_sensitive_query": is_time_sensitive_query,
                    "is_current_affairs_query": is_current_affairs_query,
                    "timings": {"analyze_query": perf_counter() - start},
                })
                logger.info("Abuse/scope gate triggered (category=%s).", abuse_category)
                return updates

        effective_query = normalized_query
        memory_intent = self._detect_memory_intent(effective_query)
        intro_entities = self._extract_intro_entities(effective_query)
        intro_ack = self._build_intro_acknowledgement(intro_entities)
        if memory_intent:
            updates["memory_intent"] = memory_intent
        if intro_entities:
            updates["captured_intro_entities"] = intro_entities

        # Baseline result derived from the orchestrator plan.
        result = QueryClassification(
            query_type=classification_query_type,
            translated_query=plan_query_en or final_query,
            confidence=plan_confidence,
            reasoning="Turn plan (orchestrator).",
            subjects=plan_subjects,
            class_level=plan.get("class_level"),
        )

        # --- Deterministic overrides applied on top of the plan ---
        if (
            (interactive_followup or has_images)
            and not wants_visual_eff
            and plan_query_type != "image_generation"
        ):
            # Keep short replies / image uploads inside the educational pipeline so
            # they don't leak into the ConversationalAgent.
            reason = (
                "Image upload: route to interactive vision pipeline."
                if has_images
                else "Interactive follow-up: short reply to open Socratic question."
            )
            result = QueryClassification(
                query_type="curriculum_specific",
                translated_query=plan_query_en or final_query,
                confidence=1.0,
                reasoning=reason,
                subjects=plan_subjects or ["General"],
            )
        elif (
            heuristic_result is not None
            and heuristic_result.query_type == "conversational"
            and not wants_visual_eff
            and plan_query_type != "image_generation"
        ):
            # Deterministic social shortcut (greeting / acknowledgment / vague help)
            # is honored even if the plan classified the turn differently.
            result = heuristic_result
        elif intro_ack and plan_query_type != "image_generation":
            result = QueryClassification(
                query_type="conversational",
                translated_query=effective_query,
                confidence=1.0,
                reasoning="Matched self-introduction heuristic.",
                subjects=["General"],
            )
            updates["response"] = intro_ack
        elif memory_intent and plan_query_type != "image_generation":
            result = QueryClassification(
                query_type="conversational",
                translated_query=effective_query,
                confidence=1.0,
                reasoning=f"Matched direct memory intent ({memory_intent}).",
                subjects=["General"],
            )

        # Final safety: an interactive follow-up must never reach the conversational agent.
        if (
            interactive_followup
            and not has_images
            and result.query_type == "conversational"
            and not wants_visual_eff
        ):
            logger.info(
                "Final safety override: forcing interactive follow-up to curriculum_specific (was '%s').",
                result.query_type,
            )
            result = QueryClassification(
                query_type="curriculum_specific",
                translated_query=result.translated_query or plan_query_en or final_query,
                confidence=1.0,
                reasoning="Interactive follow-up override (post-plan).",
                subjects=result.subjects or plan_subjects or ["General"],
            )

        # Acknowledgment detection (deterministic).
        normalized_chat = self._classifier._normalize_chat_phrase(query)
        normalized_chat_compact = re.sub(r"([a-z])\1{2,}", r"\1\1", normalized_chat)
        ack_tokens = set(getattr(self._classifier, "_ACK_TOKENS", set()) or set())
        is_acknowledgment_turn = bool(
            result.query_type == "conversational"
            and (
                normalized_chat in ack_tokens
                or normalized_chat_compact in ack_tokens
                or "acknowledgment" in (result.reasoning or "").lower()
            )
        )
        # Acknowledgments during an interactive follow-up are answers to a Socratic
        # question, not standalone acknowledgments.
        if interactive_followup:
            is_acknowledgment_turn = False

        updates.update({
            "query_en": result.translated_query,
            "query_type": result.query_type,
            "subjects": result.subjects,
            "is_acknowledgment": is_acknowledgment_turn,
            # The single LLM planning call is accounted for in orchestrate_turn.
            "llm_calls": 0,
            "is_time_sensitive_query": is_time_sensitive_query,
            "is_current_affairs_query": is_current_affairs_query,
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
        # locked preference > memory preference > high-confidence detection > English.
        min_lang_conf = float(settings.language_response_min_confidence or 0.80)
        if language_preference_locked:
            effective_language = preferred_response_language
        else:
            resolved = LanguageDetector.resolve_response_language(
                detected_lang,
                detect_confidence,
                min_confidence=min_lang_conf,
            )
            if resolved == "en" and detected_lang != "en" and detect_confidence < min_lang_conf:
                logger.info(
                    "Language confidence %.3f for '%s' below %.2f; responding in English",
                    detect_confidence,
                    detected_lang,
                    min_lang_conf,
                )
            effective_language = resolved
        if (
            not language_preference_locked
            and updates.get("is_acknowledgment")
            and self._detector._is_short_english_chat(query)
        ):
            effective_language = "en"
        if not effective_language or str(effective_language).lower() == "null":
            effective_language = "en"
        updates["preferred_response_language"] = preferred_response_language
        updates["language_preference_locked"] = language_preference_locked
        updates["language"] = effective_language

        # Persist explicit language preference in session metadata so lock survives
        # across requests even when transient graph state is not retained.
        if parsed_preferred and parsed_locked:
            current_meta = updates.get("session_metadata", state.get("session_metadata", {})) or {}
            new_meta = dict(current_meta)
            structured_raw = new_meta.get("structured_summary") or {}
            if hasattr(structured_raw, "model_dump"):
                structured = dict(structured_raw.model_dump() or {})
            elif isinstance(structured_raw, dict):
                structured = dict(structured_raw)
            else:
                structured = {}
            existing_prefs = str(structured.get("user_preferences") or "").strip()
            pref_marker = f"preferred_response_language: {parsed_preferred}"
            if pref_marker not in existing_prefs.lower():
                structured["user_preferences"] = (
                    f"{existing_prefs}; {pref_marker}".strip("; ").strip()
                    if existing_prefs else pref_marker
                )
            new_meta["structured_summary"] = structured
            updates["session_metadata"] = new_meta
        
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
        elif updates.get("captured_intro_entities"):
            current_meta = state.get("session_metadata", {})
            new_meta = current_meta.copy()
            structured = dict(new_meta.get("structured_summary") or {})
            key_entities = dict(structured.get("key_entities") or {})
            if not key_entities.get("name") and updates["captured_intro_entities"].get("name"):
                key_entities["name"] = updates["captured_intro_entities"]["name"]
            if not key_entities.get("class_level") and updates["captured_intro_entities"].get("class_level"):
                key_entities["class_level"] = updates["captured_intro_entities"]["class_level"]
            if key_entities:
                structured["key_entities"] = key_entities
                new_meta["structured_summary"] = structured
                updates["session_metadata"] = new_meta
            
            # -- SPECULATIVE RAG --
            # Trigger retrieval early if we have a translated query and subjects
            if self._retriever and result.translated_query:
                try:
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
        
        if wants_visual_eff and not has_images:
            from services.image_modality_policy import (
                conversation_supports_image_generation,
                is_vague_generation_command,
            )
            from services.image_prompt_composer import format_conversation_history

            if is_vague_generation_command(query) and conversation_supports_image_generation(history):
                ctx_block = format_conversation_history(history, max_turns=10)
                base = (updates.get("query_en") or query).strip()
                marker = "[Conversation context for image generation"
                if ctx_block and marker not in base:
                    updates["query_en"] = (
                        f"{base}\n\n"
                        f"{marker} — use full thread, not only the latest line]\n"
                        f"{ctx_block}"
                    )
                    logger.info(
                        "Enriched vague image-generation request with %d chars of chat context",
                        len(ctx_block),
                    )

        # Cache resolved query for cheap follow-up handling
        if user_session_id and result.translated_query:
            try:
                from services.cache_service import CacheService
                await CacheService.set(
                    f"resolved_query:{user_session_id}",
                    result.translated_query,
                    ttl=settings.resolved_query_cache_ttl
                )
            except Exception as e:
                logger.debug("Resolved query cache set failed: %s", e)
        
        duration = perf_counter() - start
        updates["timings"] = {"analyze_query": duration}

        findings = list(state.get("agent_findings") or [])
        findings = publish(
            findings,
            "analyze_query",
            "language",
            f"target={updates.get('language', 'en')}",
            {
                "target_lang": updates.get("language", "en"),
                "detected_lang": updates.get("detected_language", ""),
                "preference_locked": bool(updates.get("language_preference_locked")),
            },
        )
        findings = publish(
            findings,
            "analyze_query",
            "curriculum",
            f"type={updates.get('query_type', '')}",
            {
                "subjects": updates.get("subjects") or [],
                "query_en": updates.get("query_en", ""),
                "query_type": updates.get("query_type", ""),
                "class_level": (updates.get("session_metadata") or {}).get("class_level"),
            },
        )
        if updates.get("is_context_resolved_followup"):
            findings = publish(
                findings,
                "analyze_query",
                "context",
                "Session thread applied to current turn",
                {
                    "effective_query": (updates.get("effective_query") or "")[:500],
                    "wants_visual_output": bool(updates.get("wants_visual_output")),
                },
                priority=7,
            )
        updates["agent_findings"] = findings
        
        return updates
