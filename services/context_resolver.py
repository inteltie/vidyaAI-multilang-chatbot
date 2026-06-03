"""Resolve the student's current turn using full session conversation context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.image_prompt_composer import format_conversation_history


_DEICTIC_PATTERNS = (
    r"\b(this|that|it|same|above|earlier|previous|last one)\b",
    r"\bwhat we (discussed|talked about|covered)\b",
    r"\bthe (topic|concept|question|diagram|image|flowchart)\b",
    r"\bfrom (the|my) (message|upload|image)\b",
    r"\bexplain\s+it\b",
    r"\bhelp me understand\s+it\b",
)

_VISUAL_EXPLAIN_PATTERNS = (
    r"\bexplain\b.*\b(diagrammatically|diagram|flowchart|visually|picture)\b",
    r"\bdiagrammatic(al(ly)?)?\b",
    r"\bdigrammatic(al(ly)?)?\b",
    r"\b(need|want|give)\b.*\b(diagram|diagrammatic|representation|illustration|visual)\b",
    r"\b(diagrammatically|with\s+(a\s+)?flowchart)\b",
    r"\busing\s+(a\s+)?(flowchart|diagram|illustration)\b",
    r"\b(show|explain|describe)\b.*\bas\s+(a\s+)?(flowchart|diagram|chart)\b",
    r"\b(draw|sketch)\b.*\b(flowchart|diagram)\b",
    r"\bflow\s*chart\b",
    r"\bflowchat\b",
    r"\bvisuali[sz]ation\b",
    r"\bvisual\s+diagram\b",
    r"\b(asking for|want|need)\b.*\b(flow\s*chart|flowchat|visual|diagram)\b",
    r"\b(flow\s*chart|diagram|visual)\b.*\bfrom your end\b",
)

_FLOWCHART_PATTERNS = (
    r"\bflow\s*chart\b",
    r"\bflowchat\b",
    r"\bprocess\s+flow\b",
    r"\bflow\s+diagram\b",
)


@dataclass
class TurnContextResolution:
    """Output of conversation-aware query resolution for one turn."""

    effective_query: str
    context_block: str
    is_contextual_followup: bool
    wants_visual_output: bool
    wants_flowchart: bool
    response_strategy: str
    last_user_topic: str


_GOAL_PATTERNS = {
    "concept_understanding": (
        r"\b(what is|explain|understand|meaning of|how does)\b",
        "Student likely wants conceptual understanding.",
        "Start with a clear core explanation, then add one concise example.",
    ),
    "step_by_step": (
        r"\b(step by step|process|workflow|how to|procedure)\b",
        "Student likely wants procedural guidance.",
        "Respond in ordered steps with short actionable sequencing.",
    ),
    "debug_or_problem": (
        r"\b(error|issue|problem|not working|stuck|failing|bug)\b",
        "Student likely needs troubleshooting support.",
        "First diagnose likely cause, then propose the smallest corrective steps.",
    ),
    "exam_or_revision": (
        r"\b(exam|interview|revision|important|quick recap|short notes)\b",
        "Student likely wants exam-focused preparation.",
        "Prioritize high-yield points and memory-friendly structure.",
    ),
}


def _infer_response_strategy(query: str, history: Optional[List[Any]]) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "Keep response concise and directly useful."
    for _, (pattern, why, how) in _GOAL_PATTERNS.items():
        if re.search(pattern, q, re.IGNORECASE):
            return f"{why} {how}"

    last_user = _last_substantive_user_topic(history, skip_text=query) or ""
    if last_user and any(tok in q for tok in ("it", "this", "that", "same", "again")):
        return (
            "Student likely refers to prior context. Resolve the topic from previous turns "
            "before answering, then continue with focused explanation."
        )
    return "Answer directly, then add one short clarifying follow-up only if needed."


def infer_response_format(query: str) -> str | None:
    """
    Return a preferred response format if the user explicitly requests one.
    Values are normalized to: bullets | paragraph | steps
    """
    q = (query or "").strip().lower()
    if not q:
        return None
    if re.search(r"\b(bullets?|bullet\s*points?|points?|pointers?)\b", q):
        return "bullets"
    if re.search(r"\b(step\s*by\s*step|steps?|procedure)\b", q) and re.search(
        r"\b(in|as)\b", q
    ):
        return "steps"
    if re.search(r"\b(paragraph|para)\b", q):
        return "paragraph"
    return None


def wants_visual_explanation(query: str) -> bool:
    """True when the student asks for a NEW diagram/flowchart/visual to be generated."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    # Asking to explain an existing image/diagram/point is not a request to generate one.
    from services.image_modality_policy import looks_like_image_explanation_request

    if looks_like_image_explanation_request(query):
        return False
    for pattern in _VISUAL_EXPLAIN_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    return False


def wants_flowchart(query: str) -> bool:
    """True when the student asked specifically for a flowchart (not a photo-real render)."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    return any(re.search(pat, q_lower, re.IGNORECASE) for pat in _FLOWCHART_PATTERNS)


def resolve_visual_intent(
    query: str,
    conversation_history: Optional[List[Any]] = None,
) -> bool:
    """
    Decide from the full thread—not only the latest line—if we should generate a visual.

    Covers affirmations ("yes, flowchart from your end"), typos (flowchat), and prior-topic
    follow-ups where the current message is short but context makes intent clear.
    """
    current = (query or "").strip()
    from services.generated_image_context import should_explain_exploration_zone

    if should_explain_exploration_zone(current, conversation_history=conversation_history):
        return False

    if wants_visual_explanation(current):
        return True

    history = list(conversation_history or [])
    if not history:
        return False

    last_user = _last_user_message(history, skip_text=current)
    q_lower = current.lower()

    if last_user and wants_visual_explanation(last_user):
        if re.search(r"\b(yes|yeah|yep|ok|okay|sure|please|right|correct)\b", q_lower):
            return True
        if re.search(r"\b(asking for|from your end|visual|flow|diagram)\b", q_lower):
            return True

    from services.image_modality_policy import (
        conversation_supports_image_generation,
        is_vague_generation_command,
    )

    if conversation_supports_image_generation(history):
        if is_vague_generation_command(current):
            return True
        if re.search(r"\b(flow|visual|diagram|chart|image|draw|generate)\b", q_lower):
            return True

    combined = " ".join(
        _message_text(msg)
        for msg in history[-8:]
    ).strip()
    if combined and wants_visual_explanation(combined):
        if wants_visual_explanation(current) or re.search(
            r"\b(yes|visual|flow|diagram|chart|generate|draw|from your end)\b",
            q_lower,
        ):
            return True

    return False


def _message_text(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content") or msg.get("text") or "").strip()
    return str(getattr(msg, "content", "") or "").strip()


def _last_substantive_user_topic(
    history: Optional[List[Any]],
    *,
    skip_text: str = "",
) -> str:
    """Prefer the prior user turn that introduced the subject, not a visual-only line."""
    skip = (skip_text or "").strip().lower()
    for msg in reversed(history or []):
        role = ""
        if isinstance(msg, dict):
            role = str(msg.get("role") or "")
            text = str(msg.get("content") or msg.get("text") or "").strip()
        else:
            role = str(getattr(msg, "type", None) or getattr(msg, "role", "") or "")
            if role == "human":
                role = "user"
            text = str(getattr(msg, "content", "") or "").strip()
        if role not in {"user", "human"} or not text:
            continue
        if skip and text.lower() == skip:
            continue
        if wants_visual_explanation(text) and len(text) < 120:
            continue
        if len(text) >= 12:
            return text
    return _last_user_message(history, skip_text=skip_text)


def _is_deictic_or_vague_followup(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    q_lower = q.lower()
    if any(re.search(pat, q_lower) for pat in _DEICTIC_PATTERNS):
        return True
    # Very short replies that rely on prior turns (not pure ack tokens handled elsewhere).
    words = re.findall(r"[a-zA-Z']+", q_lower)
    if len(words) <= 6 and len(q) < 80:
        if wants_visual_explanation(q):
            return True
        if any(w in q_lower for w in ("explain", "more", "again", "why", "how", "what", "continue")):
            return True
    return False


def _last_user_message(history: Optional[List[Any]], *, skip_text: str = "") -> str:
    skip = (skip_text or "").strip().lower()
    for msg in reversed(history or []):
        role = ""
        if isinstance(msg, dict):
            role = str(msg.get("role") or "")
            text = str(msg.get("content") or msg.get("text") or "").strip()
        else:
            role = str(getattr(msg, "type", None) or getattr(msg, "role", "") or "")
            if role == "human":
                role = "user"
            text = str(getattr(msg, "content", "") or "").strip()
        if role in {"user", "human"} and text:
            if skip and text.lower() == skip:
                continue
            return text
    return ""


def _last_assistant_excerpt(history: Optional[List[Any]], *, max_chars: int = 400) -> str:
    for msg in reversed(history or []):
        role = ""
        if isinstance(msg, dict):
            role = str(msg.get("role") or "")
            text = str(msg.get("content") or msg.get("text") or "").strip()
        else:
            role = str(getattr(msg, "type", None) or getattr(msg, "role", "") or "")
            if role == "ai":
                role = "assistant"
            text = str(getattr(msg, "content", "") or "").strip()
        if role in {"assistant", "ai"} and text:
            return text[:max_chars]
    return ""


def resolve_turn_context(
    query: str,
    conversation_history: Optional[List[Any]],
    *,
    session_metadata: Optional[Dict[str, Any]] = None,
    max_history_turns: int = 10,
) -> TurnContextResolution:
    """
    Build an effective query and tutor context block from the full recent chat.

    Used for every curriculum turn so short or deictic messages ("explain it",
    "with a flowchart", "no just generate it") keep the thread topic.
    """
    current = (query or "").strip()
    history = list(conversation_history or [])
    history_block = format_conversation_history(history, max_turns=max_history_turns)
    last_user = _last_substantive_user_topic(history, skip_text=current) or _last_user_message(
        history, skip_text=current
    )
    last_assistant = _last_assistant_excerpt(history)
    visual = resolve_visual_intent(current, history)
    flowchart = wants_flowchart(current) or (
        bool(last_user) and wants_flowchart(last_user)
    )
    strategy = _infer_response_strategy(current, history)
    # Visual requests are explicit even when not deictic.
    is_followup = _is_deictic_or_vague_followup(current) or (
        visual and bool(history)
    )

    topic_hint = last_user or last_assistant or ""
    if session_metadata:
        structured = session_metadata.get("structured_summary") or {}
        if hasattr(structured, "model_dump"):
            structured = structured.model_dump()
        entities = (structured or {}).get("key_entities") or {}
        if isinstance(entities, dict):
            for key in ("topic", "subject", "subjects"):
                val = entities.get(key)
                if isinstance(val, str) and val.strip():
                    topic_hint = topic_hint or val.strip()
                    break

    if not is_followup or not history_block:
        return TurnContextResolution(
            effective_query=current,
            context_block=(
                "[Response strategy]\n"
                f"Current student message: {current}\n"
                f"How to respond: {strategy}"
            ),
            is_contextual_followup=False,
            wants_visual_output=visual,
            wants_flowchart=flowchart,
            response_strategy=strategy,
            last_user_topic=last_user,
        )

    lines = [
        "[Session context — read the full thread; the latest line may be incomplete on its own]",
        f"Current student message: {current}",
    ]
    if topic_hint:
        lines.append(f"Ongoing topic from prior turns: {topic_hint[:500]}")
    if visual:
        visual_kind = "flowchart" if flowchart else "diagram/illustration"
        lines.append(
            f"Student wants a generated {visual_kind} for the ongoing topic. "
            "Resolve the subject from prior messages; do not answer with text-only "
            "when they asked for a visual from you."
        )
    lines.append(f"How to respond for this turn: {strategy}")
    lines.append("Recent conversation:")
    lines.append(history_block)

    context_block = "\n".join(lines)
    # Full text for agents/memory; retrieval uses shorter analysis_query in analyze_query.
    effective = current

    return TurnContextResolution(
        effective_query=effective,
        context_block=context_block,
        is_contextual_followup=True,
        wants_visual_output=visual,
        wants_flowchart=flowchart,
        response_strategy=strategy,
        last_user_topic=last_user,
    )


__all__ = [
    "TurnContextResolution",
    "resolve_turn_context",
    "resolve_visual_intent",
    "wants_flowchart",
    "wants_visual_explanation",
]
