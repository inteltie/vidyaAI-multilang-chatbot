"""LangGraph node: translate_response."""

from __future__ import annotations

import logging
from typing import Any, Dict
from time import perf_counter

from services import Translator
from state import AgentState

logger = logging.getLogger(__name__)


class TranslateResponseNode:
    """LangGraph node: translate_response."""

    def __init__(self, translator: Translator) -> None:
        self._translator = translator

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        target_lang = state.get("language", "en")
        if not target_lang or str(target_lang).lower() == "null":
            target_lang = "en"
        response = state.get("response", "")
        
        duration = perf_counter() - start
        
        # Prepare updates
        updates = {
            "timings": {"translate_response": duration}
        }
        
        if not response:
            return updates

        # 1. Skip if already translated (e.g., by the Agent logic)
        if state.get("final_language") == target_lang:
            logger.info("Response already in target language (%s). Skipping translation.", target_lang)
            return updates

        # 2. Skip if target is English (no translation needed)
        if target_lang == "en":
            updates["final_language"] = "en"
            return updates

        # Always translate to non-English target languages
        # (Agents work in English internally, so response is always in English)

        # Always try to translate if target_lang is not English and not already marked as translated.
        logger.info("TranslateResponseNode: Ensuring response is in target language: %s", target_lang)
        translated, i_tokens, o_tokens = await self._translator.from_english(response, target_lang)
        
        # Recalculate duration after LLM call
        duration = perf_counter() - start
        updates["timings"]["translate_response"] = duration
        
        updates["response"] = translated
        updates["final_language"] = target_lang
        updates["llm_calls"] = 1
        updates["input_tokens"] = i_tokens
        updates["output_tokens"] = o_tokens

        return updates


