"""Build TutorInput package for InteractiveStudentAgent from agent findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from services.agent_findings import compose_tutor_context


@dataclass
class TutorInput:
    query_for_react: str
    system_extra: str
    turn_type: str


class TutorContextComposer:
    """Assemble tutor-facing context from the findings bus and state."""

    def compose(self, state: Dict[str, Any]) -> TutorInput:
        findings = state.get("agent_findings")
        parts = compose_tutor_context(findings, state)

        query_en = (state.get("query_en") or state.get("query") or "").strip()
        inject_block = parts.image_inject_block.strip()

        query_for_react = query_en
        if inject_block and inject_block not in query_en:
            query_for_react = f"{query_en}\n\n{inject_block}" if query_en else inject_block

        system_extra = ""
        if inject_block:
            system_extra = (
                "The student has shared image(s) in this session. "
                "Use the visual context below when guiding them.\n"
                f"{inject_block}"
            )

        timeline = list(state.get("message_modality_timeline") or [])
        # Fresh upload already has full visual context; older timeline markers confuse the tutor.
        if state.get("images") or state.get("new_uploaded_image_records"):
            timeline = []
        if timeline:
            event_lines = ["[Session message image events]"]
            for ev in timeline[-12:]:
                role = ev.get("role", "user")
                who = "Student" if role == "user" else "Tutor"
                flags = []
                if ev.get("has_image_input"):
                    flags.append("uploaded image")
                if ev.get("has_generated_image"):
                    flags.append("generated diagram")
                if flags:
                    event_lines.append(f"- {who}: {', '.join(flags)} — {ev.get('content_preview', '')[:80]}")
            if len(event_lines) > 1:
                system_extra = f"{system_extra}\n\n{chr(10).join(event_lines)}".strip()

        remembered_generated = list(state.get("remembered_generated_images") or [])
        if remembered_generated:
            generated_lines = ["[Previously generated diagrams in this session]"]
            for rec in remembered_generated[-5:]:
                generated_lines.append(
                    "- image_id={iid}; prompt={prompt}; url={url}".format(
                        iid=str(rec.get("image_id") or ""),
                        prompt=str(rec.get("prompt_used") or ""),
                        url=str(rec.get("url") or ""),
                    )
                )
            generated_block = "\n".join(generated_lines)
            system_extra = f"{system_extra}\n\n{generated_block}".strip()

        return TutorInput(
            query_for_react=query_for_react,
            system_extra=system_extra,
            turn_type=parts.turn_type,
        )
