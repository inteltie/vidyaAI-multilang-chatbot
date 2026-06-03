"""Follow-up context for numbered exploration zones on generated diagrams."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_ZONE_REF_PATTERNS = (
    re.compile(
        r"\b(?:what|explain|tell\s+me\s+about|describe)\s+(?:is|was|does)?\s*"
        r"(?:the\s+)?(?:marked\s+|labeled\s+|labelled\s+)?(?:zone|point)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:marked\s+)?(?:zone|point|number)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bzone\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(
        r"\bpoint\s*#?\s*(\d+)\b(?!\s*(?:st|nd|rd|th)\b)",
        re.IGNORECASE,
    ),
)

_NEW_IMAGE_OVERRIDE = re.compile(
    r"\b(generate|create|draw|make|regenerate|another|new|different)\b.*\b"
    r"(image|diagram|illustration|picture|flowchart)\b",
    re.IGNORECASE,
)


def parse_exploration_zone_reference(query: str) -> Optional[int]:
    """Return zone/point number if the student refers to a labeled region on a prior diagram."""
    q = (query or "").strip()
    if not q:
        return None
    for pattern in _ZONE_REF_PATTERNS:
        match = pattern.search(q)
        if match:
            try:
                zone = int(match.group(1))
                if zone >= 1:
                    return zone
            except (TypeError, ValueError):
                continue
    return None


def is_exploration_zone_followup(query: str) -> bool:
    """True when the student asks about a numbered point/zone on an existing diagram."""
    if parse_exploration_zone_reference(query) is None:
        return False
    if _NEW_IMAGE_OVERRIDE.search(query or ""):
        return False
    return True


def session_has_generated_diagram(
    *,
    remembered_generated_images: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Any]] = None,
    message_modality_timeline: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """True if this session already has at least one tutor-generated diagram."""
    if remembered_generated_images:
        return True
    for ev in message_modality_timeline or []:
        if ev.get("has_generated_image"):
            return True
    for msg in reversed(conversation_history or []):
        if isinstance(msg, dict):
            if msg.get("has_generated_image"):
                return True
            continue
        kwargs = getattr(msg, "additional_kwargs", None) or {}
        if kwargs.get("has_generated_image"):
            return True
    return False


def should_explain_exploration_zone(
    query: str,
    *,
    remembered_generated_images: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Any]] = None,
    message_modality_timeline: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Route to text explanation of a zone — do not regenerate the full image."""
    if not is_exploration_zone_followup(query):
        return False
    return session_has_generated_diagram(
        remembered_generated_images=remembered_generated_images,
        conversation_history=conversation_history,
        message_modality_timeline=message_modality_timeline,
    )


def build_exploration_zone_context_block(
    *,
    zone_id: int,
    remembered_generated_images: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build tutor context for explaining one numbered zone on the latest generated diagram."""
    records = list(remembered_generated_images or [])
    latest = records[-1] if records else {}
    zones = list(latest.get("exploration_zones") or [])
    topic = str(latest.get("prompt_used") or "").strip()

    target: Optional[Dict[str, Any]] = None
    for z in zones:
        if not isinstance(z, dict):
            continue
        if int(z.get("zone") or 0) == zone_id:
            target = z
            break

    lines = [
        "[Generated diagram follow-up — explain in text; do NOT generate a new image]",
        f"Student is asking about numbered region {zone_id} on the diagram you already created.",
        "Answer with a clear educational explanation of that part only.",
    ]
    if topic:
        lines.append(f"Diagram topic: {topic[:400]}")
    if target:
        label = str(target.get("label") or "").strip()
        region = str(target.get("region_hint") or "").strip()
        question = str(target.get("curiosity_question") or "").strip()
        if label:
            lines.append(f"Zone {zone_id} label: {label}")
        if region:
            lines.append(f"Zone {zone_id} location: {region}")
        if question:
            lines.append(f"Teaching angle: {question}")
    elif zones:
        lines.append("Known zones on the last diagram:")
        for z in zones[:12]:
            if isinstance(z, dict) and z.get("label"):
                lines.append(
                    f"  - Zone {z.get('zone')}: {z.get('label')} "
                    f"({z.get('region_hint', '')})".strip()
                )
    else:
        lines.append(
            "Zone metadata was not stored; infer from the prior diagram topic and "
            f"explain what region {zone_id} typically represents."
        )
    return "\n".join(lines)


__all__ = [
    "build_exploration_zone_context_block",
    "is_exploration_zone_followup",
    "parse_exploration_zone_reference",
    "session_has_generated_diagram",
    "should_explain_exploration_zone",
]
