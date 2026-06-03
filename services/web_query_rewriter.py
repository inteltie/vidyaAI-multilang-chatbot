"""Query rewriting service for web fallback optimization."""

from __future__ import annotations

import logging
import re
from typing import Iterable

from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)


class WebQueryRewriter:
    """Rewrites user query into concise web-search query."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return getattr(self._llm, "model_name", "unknown")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
        return {w for w in words if len(w) >= 3}

    def _guard_intent(
        self,
        original_query: str,
        rewritten_query: str,
        context_snippets: Iterable[str] | None = None,
    ) -> dict[str, object]:
        original_tokens = self._tokens(original_query)
        rewritten_tokens = self._tokens(rewritten_query)
        context_tokens: set[str] = set()
        for snippet in (context_snippets or []):
            context_tokens |= self._tokens(str(snippet))

        if not original_tokens:
            return {
                "accepted": True,
                "reason": "no_original_tokens",
                "overlap_ratio": 1.0,
            }

        overlap_with_original = len(original_tokens & rewritten_tokens)
        overlap_ratio = overlap_with_original / max(len(original_tokens), 1)
        # Balanced guard: permit context disambiguation, but prevent major intent drift.
        only_context_terms = len((rewritten_tokens - original_tokens) & context_tokens)
        accepted = overlap_ratio >= 0.35 or (overlap_ratio >= 0.2 and only_context_terms > 0)
        # Extra guard for short/ambiguous queries: if rewrite introduces many off-topic
        # tokens and keeps little from current query, force fallback.
        rewritten_non_context = len(rewritten_tokens - original_tokens - context_tokens)
        if overlap_ratio < 0.25 and rewritten_non_context >= 3:
            accepted = False
        reason = "accepted" if accepted else "low_overlap_drift"
        return {
            "accepted": accepted,
            "reason": reason,
            "overlap_ratio": round(overlap_ratio, 4),
        }

    async def rewrite(
        self,
        query_en: str,
        context_snippets: Iterable[str] | None = None,
        fallback_reason: str | None = None,
    ) -> tuple[str, int, int, dict[str, object]]:
        context = " | ".join([c.strip() for c in (context_snippets or []) if str(c).strip()])
        prompt = (
            "Rewrite the user query into a single concise web-search query.\n"
            "Rules:\n"
            "- Prioritize the CURRENT user query and preserve all entities, constraints, and intent.\n"
            "- Use context only to disambiguate short/ambiguous phrasing.\n"
            "- Do not invent facts.\n"
            "- Do not switch topic to earlier conversation turns.\n"
            "- Output only the rewritten query text.\n"
            "- Keep it short (target <=18 tokens, unless needed).\n\n"
            f"Fallback reason: {fallback_reason or 'unknown'}\n"
            f"Context: {context or 'None'}\n"
            f"Query: {query_en}\n"
        )
        try:
            resp = await self._llm.ainvoke(prompt)
            rewritten_raw = (resp.content or "").strip() or query_en
            guard = self._guard_intent(
                original_query=query_en,
                rewritten_query=rewritten_raw,
                context_snippets=context_snippets,
            )
            rewritten = rewritten_raw if bool(guard.get("accepted")) else query_en
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
            i_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            o_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            logger.info(
                "[TOKEN_USAGE] WebQueryRewriter: input_tokens=%s, output_tokens=%s, model=%s, guard=%s",
                i_tokens,
                o_tokens,
                self.model_name,
                guard,
            )
            return rewritten, i_tokens, o_tokens, guard
        except Exception as exc:
            logger.warning("Web query rewrite failed: %s", exc)
            return query_en, 0, 0, {
                "accepted": False,
                "reason": "rewrite_exception",
                "overlap_ratio": 0.0,
            }

