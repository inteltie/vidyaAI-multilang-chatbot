"""Rules-first policy for when to inject remembered image context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.messages import BaseMessage

from config import settings

InjectLevel = Literal["none", "compact", "full", "ambiguous"]
PolicyResult = Literal["none", "compact", "full", "ambiguous"]

_VISUAL_REF_PATTERNS = (
    r"\bdiagram\b",
    r"\bimage\b",
    r"\bphoto\b",
    r"\bpicture\b",
    r"\bfigure\b",
    r"\bgraph\b",
    r"\bchart\b",
    r"\bequation\b",
    r"\bthis\b",
    r"\bthat\b",
    r"\bstep\s*\d+",
    r"\bचित्र\b",
    r"\bफोटो\b",
    r"\bआरेख\b",
)


@dataclass
class ImageGateDecision:
    inject_level: InjectLevel
    selected_image_ids: List[str]
    reason: str


def _message_content(msg: Union[Dict[str, Any], BaseMessage]) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content") or msg.get("text") or "")
    return str(getattr(msg, "content", "") or "")


def _message_role(msg: Union[Dict[str, Any], BaseMessage]) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role") or "user")
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return str(role or "user")


def _has_visual_marker(text: str) -> bool:
    t = (text or "").lower()
    if "[attached images]" in t or "[image context]" in t:
        return True
    return any(re.search(pat, t) for pat in _VISUAL_REF_PATTERNS)


def _user_turns_since_visual(history: List[Any]) -> int:
    """Count user turns since last message with image context marker."""
    count = 0
    for msg in reversed(history or []):
        role = _message_role(msg)
        content = _message_content(msg)
        if role == "user":
            if _has_visual_marker(content):
                return count
            count += 1
    return count


def _last_assistant_text(history: List[Any]) -> str:
    for msg in reversed(history or []):
        if _message_role(msg) == "assistant":
            return _message_content(msg)
    return ""


def _ids_for_upload_batch(
    remembered: List[Dict[str, Any]],
    upload_id: str,
) -> List[str]:
    return [
        str(r["image_id"])
        for r in remembered
        if r.get("upload_id") == upload_id and r.get("image_id")
    ]


def evaluate_image_context_policy(
    *,
    query: str,
    has_new_images: bool,
    remembered_images: List[Dict[str, Any]],
    conversation_history: List[Any],
    new_upload_records: Optional[List[Dict[str, Any]]] = None,
    is_topic_shift: bool = False,
    turn_type: str = "opening",
) -> ImageGateDecision:
    """Decide none / compact / full / ambiguous image injection."""
    remembered = list(remembered_images or [])
    all_ids = [r.get("image_id", "") for r in remembered if r.get("image_id")]

    if has_new_images:
        new_records = list(new_upload_records or [])
        new_ids = [str(r["image_id"]) for r in new_records if r.get("image_id")]
        if not new_ids and new_records:
            upload_id = str(new_records[0].get("upload_id") or "")
            if upload_id:
                new_ids = _ids_for_upload_batch(remembered, upload_id)
        if not new_ids and remembered:
            latest_upload = remembered[-1].get("upload_id")
            if latest_upload:
                new_ids = _ids_for_upload_batch(remembered, str(latest_upload))
        return ImageGateDecision(
            inject_level="full",
            selected_image_ids=new_ids or all_ids[-3:],
            reason="New images uploaded this turn (current upload only)",
        )

    if not remembered:
        return ImageGateDecision("none", [], "No remembered images in session")

    q = (query or "").strip()
    q_lower = q.lower()

    if is_topic_shift and not _has_visual_marker(q):
        return ImageGateDecision("none", [], "Topic shift without visual reference")

    if _has_visual_marker(q):
        # Prefer most recent upload batch
        latest_upload = remembered[-1].get("upload_id") if remembered else None
        selected = [
            r["image_id"]
            for r in remembered
            if r.get("upload_id") == latest_upload and r.get("image_id")
        ] or ([remembered[-1]["image_id"]] if remembered else [])
        return ImageGateDecision(
            "full",
            selected,
            "Explicit visual reference in query",
        )

    window = settings.image_context_turn_window
    turns_since = _user_turns_since_visual(conversation_history)
    if turns_since <= window:
        recent_ids = all_ids[-3:]
        return ImageGateDecision(
            "compact",
            recent_ids,
            f"Recent visual thread ({turns_since} user turns since upload)",
        )

    if turn_type in ("followup", "confusion", "meta"):
        last_a = _last_assistant_text(conversation_history).lower()
        if any(w in last_a for w in ("diagram", "image", "figure", "graph", "equation", "photo")):
            return ImageGateDecision(
                "compact",
                all_ids[-2:],
                f"Interactive {turn_type} after visual assistant turn",
            )

    if len(q) < 20 and turns_since < window + 2:
        return ImageGateDecision("ambiguous", all_ids[-1:], "Short reply near visual context")

    return ImageGateDecision("none", [], "Conversation history sufficient")
