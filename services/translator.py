"""Translation service using the shared LLM."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class Translator:
    """Bidirectional translator using the LLM."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def to_english(self, text: str, source_lang: str) -> str:
        """Translate the given text to English, returning original on failure."""
        if source_lang == "en":
            return text
        prompt = (
            f"Translate the following {source_lang} educational text into clear English. "
            "Respond with only the translated text.\n\n"
            f"Text: {text}"
        )
        try:
            resp = await self._llm.ainvoke(prompt)
            return (resp.content or "").strip() or text
        except Exception as exc:  # pragma: no cover
            logger.warning("Translation to English failed: %s", exc)
            return text

    async def from_english(self, text: str, target_lang: str) -> str:
        """Translate an English text to the target language, returning original on failure."""
        if target_lang == "en":
            return text
        prompt = (
            f"You are a professional translator. Task: Ensure the following text is in **{target_lang}**. \n\n"
            f"1. If the text is NOT in {target_lang}, translate it ENTIRELY into {target_lang}.\n"
            f"2. If the text is ALREADY in {target_lang}, return it exactly as it is.\n\n"
            "Preserve technical terms and academic structure. Respond ONLY with the final text.\n\n"
            f"Text: {text}"
        )
        try:
            resp = await self._llm.ainvoke(prompt)
            return (resp.content or "").strip() or text
        except Exception as exc:  # pragma: no cover
            logger.warning("Translation from English failed: %s", exc)
            return text


