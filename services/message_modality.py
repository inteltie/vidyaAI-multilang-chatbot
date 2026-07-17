"""Helpers for image input/output metadata on chat messages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def empty_modality() -> Dict[str, Any]:
    return {
        "has_image_input": False,
        "image_input_ids": [],
        "image_input_urls": [],
        "has_generated_image": False,
        "generated_image_ids": [],
        "generated_image_urls": [],
    }


def modality_from_message(msg: Any) -> Dict[str, Any]:
    """Read modality flags from ChatMessage, dict buffer row, or LangChain message."""
    if msg is None:
        return empty_modality()

    if isinstance(msg, dict):
        return {
            "has_image_input": bool(msg.get("has_image_input")),
            "image_input_ids": list(msg.get("image_input_ids") or []),
            "image_input_urls": list(msg.get("image_input_urls") or []),
            "has_generated_image": bool(msg.get("has_generated_image")),
            "generated_image_ids": list(msg.get("generated_image_ids") or []),
            "generated_image_urls": list(msg.get("generated_image_urls") or []),
        }

    kwargs = getattr(msg, "additional_kwargs", None) or {}
    if kwargs:
        return {
            "has_image_input": bool(kwargs.get("has_image_input")),
            "image_input_ids": list(kwargs.get("image_input_ids") or []),
            "image_input_urls": list(kwargs.get("image_input_urls") or []),
            "has_generated_image": bool(kwargs.get("has_generated_image")),
            "generated_image_ids": list(kwargs.get("generated_image_ids") or []),
            "generated_image_urls": list(kwargs.get("generated_image_urls") or []),
        }

    return {
        "has_image_input": bool(getattr(msg, "has_image_input", False)),
        "image_input_ids": list(getattr(msg, "image_input_ids", None) or []),
        "image_input_urls": list(getattr(msg, "image_input_urls", None) or []),
        "has_generated_image": bool(getattr(msg, "has_generated_image", False)),
        "generated_image_ids": list(getattr(msg, "generated_image_ids", None) or []),
        "generated_image_urls": list(getattr(msg, "generated_image_urls", None) or []),
    }


def modality_to_kwargs(modality: Dict[str, Any]) -> Dict[str, Any]:
    """LangChain additional_kwargs payload for a message."""
    return {
        "has_image_input": bool(modality.get("has_image_input")),
        "image_input_ids": list(modality.get("image_input_ids") or []),
        "image_input_urls": list(modality.get("image_input_urls") or []),
        "has_generated_image": bool(modality.get("has_generated_image")),
        "generated_image_ids": list(modality.get("generated_image_ids") or []),
        "generated_image_urls": list(modality.get("generated_image_urls") or []),
    }


def format_modality_tag(modality: Dict[str, Any]) -> str:
    """Short tag for planners/history formatters."""
    parts: List[str] = []
    if modality.get("has_image_input"):
        urls = modality.get("image_input_urls") or []
        ids = modality.get("image_input_ids") or []
        if urls:
            parts.append(f"[student uploaded image url={urls[0]}]")
        elif ids:
            id_hint = f" ids={','.join(ids[:3])}" if ids else ""
            parts.append(f"[student uploaded image{id_hint}]")
        else:
            parts.append("[student uploaded image]")
    if modality.get("has_generated_image"):
        urls = modality.get("generated_image_urls") or []
        ids = modality.get("generated_image_ids") or []
        if urls:
            parts.append(f"[tutor generated diagram url={urls[0]}]")
        elif ids:
            parts.append(f"[tutor generated diagram id={ids[0]}]")
        else:
            parts.append("[tutor generated diagram]")
    return " ".join(parts)


def build_user_modality(state: Dict[str, Any]) -> Dict[str, Any]:
    records = list(state.get("new_uploaded_image_records") or [])
    has_input = bool(state.get("has_images") or state.get("images") or records)
    ids: List[str] = []
    urls: List[str] = []
    for rec in records:
        for key in ("upload_id", "image_id"):
            val = str(rec.get(key) or "").strip()
            if val and val not in ids:
                ids.append(val)
        url = str(rec.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
    return {
        "has_image_input": has_input,
        "image_input_ids": ids,
        "image_input_urls": urls,
        "has_generated_image": False,
        "generated_image_ids": [],
        "generated_image_urls": [],
    }


def build_assistant_modality(state: Dict[str, Any]) -> Dict[str, Any]:
    records = list(state.get("new_generated_image_records") or [])
    ids: List[str] = []
    urls: List[str] = []
    for rec in records:
        iid = str(rec.get("image_id") or "").strip()
        url = str(rec.get("url") or "").strip()
        if iid:
            ids.append(iid)
        if url:
            urls.append(url)
    return {
        "has_image_input": False,
        "image_input_ids": [],
        "image_input_urls": [],
        "has_generated_image": bool(records),
        "generated_image_ids": ids,
        "generated_image_urls": urls,
    }


def format_history_with_modality(
    history: Optional[List[Any]],
    *,
    max_turns: int = 8,
    max_chars: int = 4500,
) -> str:
    """Format chat history including image input/output markers per message."""
    if not history:
        return ""

    lines: List[str] = []
    for msg in history[-(max_turns * 2) :]:
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "")
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        text = str(content or "").strip()
        if not text:
            continue

        mod = modality_from_message(msg)
        tag = format_modality_tag(mod)
        if role in {"human", "user"}:
            label = "Student"
        elif role in {"ai", "assistant"}:
            label = "Tutor"
        else:
            continue
        prefix = f"{label} {tag}: " if tag else f"{label}: "
        lines.append(f"{prefix}{text[:900]}")

    block = "\n".join(lines)
    if len(block) > max_chars:
        return block[-max_chars:]
    return block


__all__ = [
    "build_assistant_modality",
    "build_user_modality",
    "format_history_with_modality",
    "format_modality_tag",
    "modality_from_message",
    "modality_to_kwargs",
]
