"""LangGraph node: translate_response."""

from __future__ import annotations

from time import perf_counter

from services import Translator
from state import AgentState


class TranslateResponseNode:
    """LangGraph node: translate_response."""

    def __init__(self, translator: Translator) -> None:
        self._translator = translator

    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        # Use user's preferred language (from request) as target, not valid detected language
        target_lang = state.get("language", "en")
        response = state.get("response", "")

        if not response:
            duration = perf_counter() - start
            timings = state.get("timings") or {}
            timings["translate_response"] = duration
            state["timings"] = timings
            return state

        # Check if response is already translated by the agent
        if state.get("is_translated"):
            # Already in target language, skip translation
            state["final_language"] = target_lang
            # No additional LLM call
        elif target_lang != "en":
            translated = await self._translator.from_english(response, target_lang)
            state["response"] = translated
            state["final_language"] = target_lang
            # one LLM call for response translation
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        else:
            state["final_language"] = "en"

        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["translate_response"] = duration
        state["timings"] = timings
        return state


