"""LLM-backed query understanding layer.

This runs BEFORE any routing or generation decision. Its job is *only* to
understand what the student is really asking for — using the full conversation
context, not a single message in isolation — so that downstream routing
(text vs. analyze-image vs. generate-image) and response shaping are grounded
in genuine intent rather than brittle keyword heuristics.

The layer is intentionally decision-light: it produces a structured reading of
intent. Callers combine it with policy (agent mode, uploads, safety) to make
the final decision. On any failure it returns ``None`` so callers transparently
fall back to the heuristic resolvers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


Intent = str  # see TurnPlan.intent for the allowed values


class TurnPlan(BaseModel):
    """Authoritative, context-aware plan for how to handle the current turn.

    Produced in a single LLM pass over the full conversation. It owns BOTH the
    interpretation of intent AND the routing decision (curriculum RAG vs. web
    vs. conversation vs. image). Deterministic policy (uploads, agent mode,
    safety) is applied afterwards as hard guardrails by ``RouteDecisionNode``.
    """

    intent: str = Field(
        description=(
            "Primary intent. One of: 'new_question', 'followup_question', "
            "'explain_existing_image', 'request_new_image', 'request_flowchart', "
            "'greeting', 'closure', 'chitchat', 'clarification_needed'."
        )
    )
    route: str = Field(
        default="curriculum",
        description=(
            "The handling lane for THIS turn. One of: "
            "'conversational' (greetings, chitchat, acknowledgments, closures, meta-help), "
            "'curriculum' (an educational/academic question answerable from course material), "
            "'web' (needs fresh/real-time external info: news, latest, current events), "
            "'hybrid' (educational AND needs fresh external info), "
            "'image_generate' (the student wants a NEW diagram/visual created), "
            "'image_analyze' (the student is asking about an image/diagram)."
        ),
    )
    query_type: str = Field(
        default="curriculum_specific",
        description=(
            "Coarse category used by downstream agents. One of: "
            "'conversational', 'curriculum_specific', 'image_generation'. "
            "Do NOT use 'image_analyze' here — set route='image_analyze' instead."
        ),
    )
    needs_curriculum: bool = Field(
        default=True,
        description="True if answering well requires retrieving course/curriculum documents (RAG).",
    )
    needs_web: bool = Field(
        default=False,
        description="True if the answer depends on fresh/real-time information from the web.",
    )
    is_time_sensitive: bool = Field(
        default=False,
        description="True if the question is about recent/current/time-bound information.",
    )
    standalone_query_en: str = Field(
        default="",
        description=(
            "The fully self-contained, pronoun-resolved question in ENGLISH, ready for "
            "retrieval. Expand deictic/short messages ('why?', 'explain it') using the "
            "thread (e.g. 'Why does photosynthesis happen?'). Translate non-English input. "
            "For greetings/chitchat, echo the original message."
        ),
    )
    subjects: list[str] = Field(
        default_factory=list,
        description="Detected academic subjects (e.g. ['Physics']). Use ['General'] if unclear.",
    )
    class_level: Optional[str] = Field(
        default=None, description="Class/grade level if mentioned, e.g. '10', 'Class 9'."
    )
    resolved_topic: str = Field(
        default="",
        description=(
            "The real subject the student is asking about, resolved from the full "
            "thread. For deictic/short messages ('explain it', 'and point 1?'), infer "
            "the topic from earlier turns. Empty only for greetings/chitchat."
        ),
    )
    is_followup: bool = Field(
        default=False,
        description="True if this message continues an earlier topic rather than starting fresh.",
    )
    wants_new_image: bool = Field(
        default=False,
        description="True ONLY if the student wants a brand-new image/diagram generated.",
    )
    wants_image_explanation: bool = Field(
        default=False,
        description=(
            "True if the student wants an EXISTING image/diagram (or a numbered point "
            "on it) explained in words — NOT a new image."
        ),
    )
    wants_flowchart: bool = Field(
        default=False,
        description="True if the requested new visual should specifically be a flowchart.",
    )
    referenced_image_zone: Optional[int] = Field(
        default=None,
        description="The numbered exploration zone/point referenced, if any (e.g. 'point 2' -> 2).",
    )
    preferred_response_format: Optional[str] = Field(
        default=None,
        description="If the student asked for a specific format: 'bullets', 'paragraph', or 'steps'.",
    )
    response_strategy: str = Field(
        default="",
        description="One concise sentence on how best to respond to genuinely help the student.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True only if the request is too ambiguous to act on without asking.",
    )
    confidence: float = Field(
        default=0.0, description="Confidence in this interpretation, 0.0-1.0."
    )
    reasoning: str = Field(
        default="", description="Brief justification grounded in the conversation."
    )


_SYSTEM_PROMPT = (
    "You are the planning layer of an educational tutor. Read the ENTIRE conversation "
    "and decide how to handle the student's CURRENT turn before any answer or image is "
    "produced. You output a structured plan; downstream code applies safety/mode "
    "guardrails. Decide from genuine intent in context, never from isolated keywords.\n\n"
    "Think about context, not just the latest line:\n"
    "- Short or deictic messages ('explain it', 'and point 1?', 'do it', 'yes please') "
    "depend on earlier turns — resolve the real topic from history.\n\n"
    "DECIDE THE ROUTE (the 'route' field) and a matching 'query_type':\n"
    "- Greetings, thanks, acknowledgments, goodbyes, closures, vague meta-help "
    "('hi', 'ok', 'thanks', 'can you help me') => route='conversational', "
    "query_type='conversational', needs_curriculum=false.\n"
    "- A genuine academic/course question (concepts, problems, explanations) => "
    "route='curriculum', query_type='curriculum_specific', needs_curriculum=true.\n"
    "- Needs fresh/real-time facts (news, latest, today, current events, this year) => "
    "set is_time_sensitive=true and needs_web=true. If it is ALSO an academic question "
    "use route='hybrid'; if it is purely a freshness lookup use route='web'.\n"
    "- Wants a NEW diagram/visual created => route='image_generate', "
    "query_type='image_generation', wants_new_image=true.\n"
    "- Asking ABOUT an existing/attached image or a numbered point on it => "
    "route='image_analyze', query_type='curriculum_specific', "
    "wants_image_explanation=true, wants_new_image=false.\n\n"
    "Distinguish carefully between wanting a NEW image vs. wanting an EXISTING image "
    "explained:\n"
    "  * 'draw/generate/create/make a diagram of X', 'show me X visually' => request_new_image "
    "(wants_new_image=true, route='image_generate').\n"
    "  * 'explain the diagram', 'what is the marked point 1', 'describe this image', "
    "'what does the figure show' => explain_existing_image (wants_image_explanation=true, "
    "wants_new_image=false, route='image_analyze').\n"
    "  * 'explain photosynthesis with a diagram' wants a NEW diagram as the medium => request_new_image.\n"
    "- A flowchart request sets wants_flowchart=true and intent='request_flowchart'.\n"
    "- standalone_query_en: the self-contained, pronoun-resolved question in English, ready "
    "for retrieval. For greetings/chitchat, echo the original message.\n"
    "- subjects: detected academic subjects (['General'] if unclear); class_level if stated.\n"
    "- Capture an explicit format request (bullets / paragraph / steps) in preferred_response_format.\n"
    "- response_strategy: one sentence on how to genuinely help (depth, examples, step-by-step, etc.).\n"
    "Never invent a desire for an image the student did not ask for. When in doubt between "
    "new vs. explain, prefer explain_existing_image if a diagram already exists this session."
)


class QueryUnderstandingService:
    """Wraps an LLM to produce a :class:`TurnPlan` for each turn."""

    def __init__(self, llm: Any, *, timeout_s: float = 12.0, max_history_turns: int = 10) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_history_turns = max_history_turns
        try:
            self._structured = llm.with_structured_output(
                TurnPlan, include_raw=True
            )
        except Exception:  # pragma: no cover - defensive
            self._structured = llm.with_structured_output(TurnPlan)

    @property
    def model_name(self) -> str:
        return getattr(self._llm, "model_name", None) or "unknown"

    def _build_prompt(
        self,
        query: str,
        *,
        history_block: str,
        agent_mode: str,
        has_uploaded_images: bool,
        session_has_generated_diagram: bool,
        topic_hint: str,
    ) -> str:
        parts: List[str] = []
        parts.append(f"Agent mode: {agent_mode or 'standard'}")
        parts.append(
            f"Student uploaded an image on THIS turn: {'yes' if has_uploaded_images else 'no'}"
        )
        parts.append(
            "A diagram was already generated earlier this session: "
            f"{'yes' if session_has_generated_diagram else 'no'}"
        )
        if topic_hint:
            parts.append(f"Heuristic topic hint (may be wrong): {topic_hint[:300]}")
        if history_block:
            parts.append("\nRecent conversation:\n" + history_block)
        parts.append(f"\nCurrent student message:\n{query}")
        parts.append(
            "\nReturn the structured understanding of THIS turn."
        )
        return "\n".join(parts)

    async def understand(
        self,
        query: str,
        *,
        conversation_history: Optional[List[Any]] = None,
        agent_mode: str = "standard",
        has_uploaded_images: bool = False,
        session_has_generated_diagram: bool = False,
        topic_hint: str = "",
    ) -> Tuple[Optional[TurnPlan], Dict[str, int]]:
        """Return (plan, token_usage). plan is None on failure."""
        usage: Dict[str, int] = {"input": 0, "output": 0}
        q = (query or "").strip()
        if not q:
            return None, usage

        try:
            from services.image_prompt_composer import format_conversation_history

            history_block = format_conversation_history(
                conversation_history, max_turns=self._max_history_turns
            )
        except Exception:
            history_block = ""

        prompt = self._build_prompt(
            q,
            history_block=history_block,
            agent_mode=agent_mode,
            has_uploaded_images=has_uploaded_images,
            session_has_generated_diagram=session_has_generated_diagram,
            topic_hint=topic_hint,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await asyncio.wait_for(
                self._structured.ainvoke(messages), timeout=self._timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning("Query understanding timed out after %ss", self._timeout_s)
            return None, usage
        except Exception as exc:
            logger.warning("Query understanding failed: %s", exc)
            return None, usage

        understanding, raw = self._unpack(result)
        usage = self._extract_usage(raw)
        return understanding, usage

    @staticmethod
    def _unpack(result: Any) -> Tuple[Optional[TurnPlan], Any]:
        # include_raw=True yields {"parsed": ..., "raw": ...}; otherwise the model directly.
        if isinstance(result, dict):
            parsed = result.get("parsed")
            raw = result.get("raw")
            if isinstance(parsed, TurnPlan):
                return parsed, raw
            if isinstance(parsed, dict):
                try:
                    return TurnPlan(**parsed), raw
                except Exception:
                    return None, raw
            return None, raw
        if isinstance(result, TurnPlan):
            return result, None
        return None, None

    @staticmethod
    def _extract_usage(raw: Any) -> Dict[str, int]:
        usage = {"input": 0, "output": 0}
        if raw is None:
            return usage
        meta = getattr(raw, "usage_metadata", None)
        if not meta:
            meta = (getattr(raw, "response_metadata", {}) or {}).get("token_usage", {})
        if isinstance(meta, dict):
            usage["input"] = int(
                meta.get("input_tokens") or meta.get("prompt_tokens") or 0
            )
            usage["output"] = int(
                meta.get("output_tokens") or meta.get("completion_tokens") or 0
            )
        return usage


# Backwards-compatible alias: the model was historically named QueryUnderstanding.
QueryUnderstanding = TurnPlan


__all__ = ["TurnPlan", "QueryUnderstanding", "QueryUnderstandingService"]
