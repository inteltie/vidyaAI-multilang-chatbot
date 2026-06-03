"""Detect whether a request should analyze uploaded images or generate new ones."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.messages import BaseMessage

from config import settings

ImageModality = Literal["none", "analyze", "generate"]

_VISUAL_NOUN = r"(diagram|image|picture|illustration|flowchart|chart|graph|visual)"

_GENERATE_PATTERNS = (
    r"\b(generate|create|draw|make|sketch|illustrate)\b.*\b"
    + _VISUAL_NOUN
    + r"\b",
    r"\b(diagram|flowchart|illustration)\b.*\b(for me|please)\b",
    r"\bdraw\b.*\b(me|a)\b",
    r"\bmake\s+(a|an)\s+(diagram|flowchart|chart|illustration)\b",
    r"\bshow\b.*\b(with|as)\b.*\b(image|diagram|picture|chart)\b",
    r"\bcan you\b.*\b(draw|generate|create|show)\b.*\b(image|diagram|chart|picture|flowchart)\b",
    r"\b(explain|describe|teach)\b.*\b(with|using|via)\b.*\b(help\s+of\s+)?(a\s+)?"
    + _VISUAL_NOUN
    + r"\b",
    r"\b(with|using)\b.*\b(help\s+of\s+)?(a\s+)?" + _VISUAL_NOUN + r"\b",
    r"\bhelp\s+of\s+(a\s+)?" + _VISUAL_NOUN + r"\b",
    r"\bvisuali[sz]e\b.*\b(this|it|concept|topic)\b",
    r"\bबनाओ\b.*\b(चित्र|आरेख|डायग्राम)\b",
    r"\bचित्र\b.*\b(बनाओ|बनाएं|जनरेट)\b",
    r"\bआरेख\b.*\b(बनाओ|बनाएं)\b",
    r"\bdia?gramm?atic(al(ly)?)?\b",
    r"\b(need|want)\b.*\b(diagrammatic|representation)\b.*\b(diagram|illustration|image)?\b",
)

_VISUAL_WORDS = frozenset(
    {
        "image",
        "diagram",
        "diagrammatic",
        "digrammatic",
        "picture",
        "flowchart",
        "illustration",
        "chart",
        "graph",
        "visual",
        "representation",
    }
)
_GENERATE_VERBS = frozenset(
    {"show", "create", "draw", "generate", "explain", "illustrate", "visualize", "visualise"}
)

_REFUSAL_PATTERNS = (
    r"\b(nude|naked|nsfw|porn|violent|gore)\b",
    r"\bexam\s+answers?\b",
    r"\bcheat\b",
)

# Explicit creation verbs: even if "explain" appears, these still mean generate.
_EXPLICIT_CREATE = re.compile(
    r"\b(generate|create|draw|sketch|illustrate|re-?draw|re-?generate|render|produce|design|"
    r"make\s+(me\s+)?(a|an|the)\b|give\s+me\s+(a|an|the)\b)\b",
    re.IGNORECASE,
)

# References to an EXISTING image/diagram the student wants explained (not a new one).
_EXPLAIN_EXISTING_VISUAL = re.compile(
    r"\b(explain|describe|elaborate|interpret|understand|clarify|"
    r"tell\s+me\s+about|what(?:'s|\s+is|\s+are|\s+does)|break\s+down|walk\s+me\s+through|read)\b"
    r".{0,48}?"
    r"\b(the|this|that|above|attached|shared|generated|previous|prior|marked|"
    r"labell?ed|same|first|second|last)\b"
    r".{0,24}?"
    r"\b(image|diagram|picture|illustration|figure|flow\s*chart|chart|visual|graphic|"
    r"point|zone|label|part|region|step|stage|component|section)s?\b",
    re.IGNORECASE | re.DOTALL,
)


def looks_like_image_explanation_request(query: str) -> bool:
    """
    True when the student wants an EXISTING image/diagram (or a numbered point on it)
    explained in text — not a new image generated.
    """
    q = (query or "").strip()
    if not q:
        return False
    if _EXPLICIT_CREATE.search(q):
        return False
    if _EXPLAIN_EXISTING_VISUAL.search(q):
        return True
    from services.generated_image_context import parse_exploration_zone_reference

    return parse_exploration_zone_reference(q) is not None

_SHORT_GENERATION_PATTERNS = (
    r"\bjust\s+generate\b",
    r"\bjust\s+generate\s+(the\s+)?(image|diagram|illustration|picture)\b",
    r"\bgenerate\s+(the\s+)?(image|diagram|illustration|picture)\b",
    r"\bgenerate\s+it\b",
    r"\b(no|yes),?\s+just\s+generate\b",
    r"\bplease\s+generate\b",
    r"\bcan you just\s+(generate|draw|create)\b",
    r"\bdraw\s+it\b",
    r"\bmake\s+the\s+(diagram|image|illustration|flowchart)\b",
    r"\bcreate\s+the\s+(diagram|image|illustration)\b",
)

_CONTEXTUAL_APPROVAL_PATTERNS = (
    r"^\s*(yes|yeah|yep|ok|okay|sure|go ahead|do it|please)\s*\.?\s*$",
    r"^\s*no,?\s+(just\s+)?(generate|draw|create)\b",
)

_ASSISTANT_OFFERED_VISUAL = re.compile(
    r"\b(create|draw|generate|make|build)\b.{0,80}\b("
    + _VISUAL_NOUN
    + r"|customer journey|illustration)\b",
    re.IGNORECASE | re.DOTALL,
)

_GENERATION_CONTEXT_WORDS = frozenset(
    {
        "diagram",
        "image",
        "picture",
        "illustration",
        "flowchart",
        "chart",
        "graph",
        "visual",
        "draw",
        "generate",
        "sketch",
    }
)


def _message_text(msg: Union[Dict[str, Any], BaseMessage]) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content") or msg.get("text") or "")
    return str(getattr(msg, "content", "") or "")


def conversation_supports_image_generation(
    history: Optional[List[Any]],
    *,
    max_turns: int = 12,
) -> bool:
    """True when recent chat (whole thread slice) discusses creating/drawing a visual."""
    recent = list(history or [])[-max_turns:]
    combined = " ".join(_message_text(msg) for msg in recent).strip()
    if combined and looks_like_image_generation_query(combined):
        return True
    if combined and _ASSISTANT_OFFERED_VISUAL.search(combined):
        return True

    for msg in recent:
        text = _message_text(msg).lower()
        if not text:
            continue
        if looks_like_image_generation_query(text):
            return True
        if _ASSISTANT_OFFERED_VISUAL.search(text):
            return True
        if any(word in text for word in _GENERATION_CONTEXT_WORDS) and re.search(
            r"\b(generate|draw|create|diagram|image|illustration|flowchart|visual|sketch)\b",
            text,
        ):
            return True
        if "simplified image" in text or "customer journey" in text:
            if re.search(r"\b(image|diagram|illustration|visual|draw|generate)\b", text):
                return True
    return False


def is_vague_generation_command(query: str) -> bool:
    """True when the current message only affirms/deferrals generation to context."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    if looks_like_image_generation_followup(query):
        return True
    for pattern in _CONTEXTUAL_APPROVAL_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    if re.search(r"\b(this|it|that)\b", q_lower) and re.search(
        r"\b(generate|draw|create|image|diagram)\b", q_lower
    ):
        return True
    return False


def is_explicit_image_generation_request(query: str) -> bool:
    """True when the current message clearly asks to generate/draw an image."""
    return looks_like_image_generation_query(query) or looks_like_image_generation_followup(
        query
    )


def is_contextual_image_generation_request(
    query: str,
    conversation_history: Optional[List[Any]],
) -> bool:
    """
    Use the full recent conversation—not only the latest message—to detect
    when the student wants an illustration generated from prior context.
    """
    if looks_like_image_generation_query(query):
        return True
    if looks_like_image_generation_followup(query):
        return True
    if not conversation_supports_image_generation(conversation_history):
        return False
    if is_vague_generation_command(query):
        return True
    return False


def looks_like_image_generation_followup(query: str) -> bool:
    """Short affirmations to proceed with generation (e.g. 'no just generate it')."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    for pattern in _SHORT_GENERATION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    return False


def looks_like_image_generation_query(query: str) -> bool:
    """True when query text matches generation patterns (ignores agent_mode)."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    # Asking to explain an existing image/point is NOT a generation request.
    if looks_like_image_explanation_request(query):
        return False
    for pattern in _GENERATE_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    tokens = set(re.findall(r"[a-z]+", q_lower))
    if tokens & _VISUAL_WORDS and tokens & _GENERATE_VERBS:
        return True
    if "help of" in q_lower and tokens & _VISUAL_WORDS:
        return True
    return False


def evaluate_image_modality(
    query: str,
    *,
    has_images: bool,
    agent_mode: str,
    conversation_history: Optional[List[Any]] = None,
    wants_visual_output: bool = False,
    remembered_generated_images: Optional[List[Any]] = None,
    message_modality_timeline: Optional[List[Any]] = None,
    suppress_generation: bool = False,
) -> ImageModality:
    """Return analyze, generate, or none for routing."""
    if has_images:
        return "analyze"

    mode = (agent_mode or "standard").strip().lower()
    if mode != "interactive":
        return "none"

    if not settings.image_generation_enabled:
        return "none"

    from services.generated_image_context import should_explain_exploration_zone

    if suppress_generation or should_explain_exploration_zone(
        query,
        remembered_generated_images=remembered_generated_images,  # type: ignore[arg-type]
        conversation_history=conversation_history,
        message_modality_timeline=message_modality_timeline,  # type: ignore[arg-type]
    ):
        return "none"

    # Explaining an existing image/diagram (e.g. "explain the diagram", "what is the image")
    # is text work, not generation — unless an explicit create verb is present.
    if looks_like_image_explanation_request(query):
        return "none"

    from services.context_resolver import resolve_visual_intent

    if wants_visual_output or resolve_visual_intent(query, conversation_history):
        return "generate"

    if is_contextual_image_generation_request(query, conversation_history):
        return "generate"

    return "none"


INTERACTIVE_IMAGE_RECOMMENDATION_SUFFIX = (
    "\n\nIf you would like me to generate a diagram for this topic, "
    "switch to interactive mode and ask me to draw it."
)


def append_interactive_image_recommendation(response: str, state: dict) -> str:
    """Append a soft interactive-mode recommendation when generation was deferred."""
    if state.get("route_fallback_reason") != "interactive_mode_required_for_image_generation":
        return response
    suffix = INTERACTIVE_IMAGE_RECOMMENDATION_SUFFIX
    if suffix.strip() in (response or ""):
        return response
    return (response or "").rstrip() + suffix


def is_image_generation_blocked(query: str) -> tuple[bool, str]:
    """Return (blocked, reason) for safety policy before calling image_generation."""
    q_lower = (query or "").lower()
    for pattern in _REFUSAL_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True, "Request is not allowed for educational image generation."
    return False, ""
