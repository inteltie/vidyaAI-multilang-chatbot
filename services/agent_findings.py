"""Shared agent findings bus for cross-agent context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from state import AgentFinding


def publish(
    findings: Optional[List[AgentFinding]],
    source: str,
    kind: str,
    summary: str,
    payload: Optional[Dict[str, Any]] = None,
    priority: int = 0,
) -> List[AgentFinding]:
    """Append a finding and return the updated list."""
    row: AgentFinding = {
        "source": source,
        "kind": kind,
        "summary": summary,
        "payload": dict(payload or {}),
        "priority": priority,
    }
    return list(findings or []) + [row]


def get_by_kind(findings: Optional[List[AgentFinding]], kind: str) -> List[AgentFinding]:
    return [f for f in (findings or []) if f.get("kind") == kind]


def get_latest_by_kind(findings: Optional[List[AgentFinding]], kind: str) -> Optional[AgentFinding]:
    matches = get_by_kind(findings, kind)
    if not matches:
        return None
    return max(matches, key=lambda f: int(f.get("priority") or 0))


def payload_for(findings: Optional[List[AgentFinding]], kind: str) -> Dict[str, Any]:
    latest = get_latest_by_kind(findings, kind)
    if not latest:
        return {}
    return dict(latest.get("payload") or {})


@dataclass
class TranslationContext:
    target_lang: str = "en"
    detected_lang: str = "en"
    subjects: List[str] = field(default_factory=list)
    student_grade: str = "B"
    visual_summary: str = ""
    validation_feedback: str = ""
    preference_locked: bool = False


@dataclass
class TutorContextParts:
    image_inject_block: str = ""
    inject_level: str = "none"
    subjects: List[str] = field(default_factory=list)
    turn_type: str = "opening"


def compose_translation_context(
    findings: Optional[List[AgentFinding]],
    *,
    target_lang: str,
    detected_lang: str = "",
    preference_locked: bool = False,
) -> TranslationContext:
    lang_payload = payload_for(findings, "language")
    curriculum = payload_for(findings, "curriculum")
    pedagogy = payload_for(findings, "pedagogy")
    visual = payload_for(findings, "visual")
    validation = payload_for(findings, "validation")

    subjects = curriculum.get("subjects") or []
    if isinstance(subjects, str):
        subjects = [subjects]

    visual_summary = ""
    if visual.get("summary_line"):
        visual_summary = str(visual["summary_line"])
    elif visual.get("image_context"):
        visual_summary = str(visual["image_context"])[:300]

    return TranslationContext(
        target_lang=target_lang or lang_payload.get("target_lang") or "en",
        detected_lang=detected_lang or lang_payload.get("detected_lang") or "",
        subjects=list(subjects),
        student_grade=str(pedagogy.get("student_grade") or "B"),
        visual_summary=visual_summary,
        validation_feedback=str(validation.get("feedback") or ""),
        preference_locked=bool(preference_locked or lang_payload.get("preference_locked")),
    )


def compose_tutor_context(findings: Optional[List[AgentFinding]], state: Dict[str, Any]) -> TutorContextParts:
    policy = payload_for(findings, "visual_policy")
    curriculum = payload_for(findings, "curriculum")
    pedagogy = payload_for(findings, "pedagogy")

    inject_level = policy.get("inject_level") or state.get("image_inject_level") or "none"
    block = policy.get("image_inject_block") or state.get("image_inject_block") or ""

    subjects = curriculum.get("subjects") or state.get("subjects") or []
    if isinstance(subjects, str):
        subjects = [subjects]

    return TutorContextParts(
        image_inject_block=str(block),
        inject_level=str(inject_level),
        subjects=list(subjects),
        turn_type=str(pedagogy.get("turn_type") or state.get("interactive_turn_type") or "opening"),
    )


def build_image_inject_block(
    inject_level: str,
    remembered: List[Dict[str, Any]],
    selected_ids: Optional[List[str]] = None,
    current_visual: Optional[Dict[str, Any]] = None,
) -> str:
    """Format image context for system prompt from registry records."""
    if inject_level == "none":
        return ""

    selected = set(selected_ids or [])
    records = list(remembered or [])

    if selected:
        records = [r for r in records if r.get("image_id") in selected]

    if not records and current_visual:
        if inject_level == "full":
            return f"[Session images]\n{current_visual.get('image_context', '')}"
        return f"[Session images]\n{current_visual.get('summary_line', '')}"

    if not records:
        return ""

    lines = ["[Session images — use only the entries below for this turn]"]
    for rec in records:
        iid = rec.get("image_id", "")
        upload_id = rec.get("upload_id", "")
        label = f"Image ({iid})" + (f" upload={upload_id}" if upload_id else "")
        if inject_level == "full":
            ctx = rec.get("image_context") or rec.get("summary_line") or ""
            lines.append(f"{label}: {ctx}")
        else:
            lines.append(f"{label}: {rec.get('summary_line', '')}")

    return "\n".join(lines)
