"""LangGraph node: parse_session_context."""

from __future__ import annotations

from time import perf_counter

from services import ContextParser
from state import AgentState


class ParseSessionContextNode:
    """LangGraph node: parse_session_context.

    Extracts session metadata (class_level, subject, chapter, last_topic)
    from conversation history.
    """

    def __init__(self, parser: ContextParser) -> None:
        self._parser = parser

    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        
        history = state.get("conversation_history", [])
        query = state.get("query", "").lower()
        
        # Heuristic: Only parse context if keywords present or history is very short
        # AND if the query is NOT conversational
        keywords = {"class", "subject", "chapter", "topic", "module", "grade", "standard", "batch"}
        query_type = state.get("query_type", "curriculum_specific")
        
        should_parse = (
            query_type != "conversational" and (
                len(history) < 2 or 
                any(k in query for k in keywords) or
                state.get("is_context_reply", False)
            )
        )
        
        if should_parse:
            extracted_metadata = await self._parser.extract_from_history(history)
            
            # MERGE extracted metadata with existing state["session_metadata"]
            # UI/existing metadata takes precedence
            current_metadata = state.get("session_metadata", {})
            
            # Map chapter -> topics if needed for consistency with DB schema
            if extracted_metadata.get("chapter") and not extracted_metadata.get("topics"):
                extracted_metadata["topics"] = extracted_metadata.pop("chapter")

            # Deterministic merge: Extracted values serve as fallbacks only if current is empty
            merged_metadata = extracted_metadata.copy()
            for k, v in current_metadata.items():
                if v: # Only override if value is present in UI/session
                    merged_metadata[k] = v
            
            state["session_metadata"] = merged_metadata
            
            # One LLM call for parsing session context
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        else:
            # Reuse existing metadata if available
            state["session_metadata"] = state.get("session_metadata", {})
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["parse_session_context"] = duration
        state["timings"] = timings
        
        return state


__all__ = ["ParseSessionContextNode"]

