"""Compose contextual, grade-appropriate image generation plans from chat state."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_openai import ChatOpenAI

from config import settings
from services.prompt_policy import GLOBAL_GUARDRAIL_RULES

logger = logging.getLogger(__name__)

ImagePlanAction = Literal["generate_diagram", "respond_text_only", "ask_clarification"]

_VAGUE_REQUEST = re.compile(
    r"\b(this|it|that|same|above|with (?:a )?(?:diagram|image|picture|chart|flowchart))\b",
    re.IGNORECASE,
)

_EDUCATIONAL_AUDIENCE = (
    "Audience: school and college students across all subjects (science, mathematics, "
    "social studies, languages, arts, commerce, etc.). "
    "Match visual style and terminology to the topic and grade level."
)

_GRADE_STYLE: Dict[str, str] = {
    "A": (
        "Advanced level; use precise domain terminology, notation, and labels appropriate "
        "to the subject (formulas, dates, anatomical terms, literary devices, etc.)."
    ),
    "B": (
        "Standard secondary/college level; label every major part, step, or concept; "
        "use correct subject vocabulary for the topic at hand."
    ),
    "C": (
        "Intermediate level; label all visible elements with clear terms and brief "
        "functional notes where helpful."
    ),
    "D": (
        "Foundational level; use simple, accurate labels with brief glosses for specialized terms."
    ),
}


@dataclass
class ImageExplorationZone:
    """One numbered region on the illustration for curiosity-driven exploration."""

    zone: int
    label: str
    region_hint: str = ""
    curiosity_question: str = ""


@dataclass
class ImageGenerationBrief:
    """Structured specification passed to the image model."""

    topic: str
    diagram_type: str = "realistic_educational_illustration"
    title: str = ""
    scene_description: str = ""
    key_elements: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    labels_required: List[str] = field(default_factory=list)
    exploration_zones: List[ImageExplorationZone] = field(default_factory=list)
    interactive_title: str = ""
    style_notes: str = ""
    avoid: List[str] = field(default_factory=list)
    source: str = "fallback"


@dataclass
class ImageGenerationPlan:
    """Agent decision for how to react to an image-related student request."""

    reasoning: str
    action: ImagePlanAction
    student_response: str
    brief: Optional[ImageGenerationBrief] = None
    source: str = "fallback"
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""


@dataclass
class ComposedImagePrompt:
    """Legacy wrapper kept for compatibility."""

    brief: ImageGenerationBrief
    prompt_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""


def _message_role(msg: Any) -> str:
    role = getattr(msg, "type", None)
    if role:
        return str(role)
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "")
    return ""


def _message_content(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    return str(content or "").strip()


def format_conversation_history(
    history: Optional[List[Any]],
    *,
    max_turns: int = 8,
    max_chars: int = 4500,
) -> str:
    """Format recent chat turns for the planning agent."""
    if not history:
        return ""

    recent = history[-(max_turns * 2) :]
    lines: List[str] = []
    for msg in recent:
        role = _message_role(msg)
        text = _message_content(msg)
        if not text:
            continue
        if role in {"human", "user"}:
            label = "Student"
        elif role in {"ai", "assistant"}:
            label = "Tutor"
        else:
            continue
        lines.append(f"{label}: {text[:900]}")

    block = "\n".join(lines)
    if len(block) > max_chars:
        return block[-max_chars:]
    return block


def _session_topic(session_metadata: Dict[str, Any]) -> str:
    structured = (session_metadata or {}).get("structured_summary") or {}
    if hasattr(structured, "model_dump"):
        structured = structured.model_dump()
    entities = dict((structured or {}).get("key_entities") or {})
    for key in ("topic", "subject", "subjects"):
        value = entities.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = ", ".join(str(v).strip() for v in value if str(v).strip())
            if joined:
                return joined
    return ""


def _last_assistant_excerpt(history: Optional[List[Any]], *, max_chars: int = 700) -> str:
    for msg in reversed(history or []):
        if _message_role(msg) in {"ai", "assistant"}:
            text = _message_content(msg)
            if text:
                return text[:max_chars]
    return ""


def _parse_exploration_zones(raw: Union[List[Any], Any]) -> List[ImageExplorationZone]:
    if not isinstance(raw, list):
        return []
    zones: List[ImageExplorationZone] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        zone_num = int(item.get("zone") or item.get("zone_id") or len(zones) + 1)
        zones.append(
            ImageExplorationZone(
                zone=zone_num,
                label=label,
                region_hint=str(item.get("region_hint") or item.get("region") or "").strip(),
                curiosity_question=str(
                    item.get("curiosity_question") or item.get("question") or ""
                ).strip(),
            )
        )
    return zones


def exploration_zones_to_dicts(zones: List[ImageExplorationZone]) -> List[Dict[str, Any]]:
    return [
        {
            "zone": z.zone,
            "label": z.label,
            "region_hint": z.region_hint,
            "curiosity_question": z.curiosity_question,
        }
        for z in zones
    ]


def build_fallback_exploration_zones(brief: ImageGenerationBrief) -> List[ImageExplorationZone]:
    """Derive numbered exploration zones from key elements when the planner omits them."""
    seeds = brief.key_elements or brief.labels_required[:8] or [brief.topic]
    zones: List[ImageExplorationZone] = []
    for idx, name in enumerate(seeds[:8], start=1):
        label = str(name).strip()
        if not label:
            continue
        zones.append(
            ImageExplorationZone(
                zone=idx,
                label=label[:60],
                region_hint=f"region associated with {label[:40]}",
                curiosity_question=(
                    f"What role does {label[:40]} play here, and what would change if it failed?"
                ),
            )
        )
    return zones


def enrich_student_response_for_exploration(
    student_response: str,
    zones: List[ImageExplorationZone],
) -> str:
    """Return planner text as-is (zones are for follow-up in chat, not UI click hints)."""
    _ = zones
    return (student_response or "").strip()


def format_brief_as_prompt(brief: ImageGenerationBrief) -> str:
    """Render a structured brief as a model-ready specification block."""
    lines = [
        f"Topic: {brief.topic}",
        f"Visual type: {brief.diagram_type or 'realistic_educational_illustration'}",
        "Audience: students (all subjects; match style and depth to the topic).",
        "Render style: clear, detailed educational illustration with accurate labels "
        "(diagram, map, timeline, process view, or realistic scene as appropriate).",
    ]
    if brief.title:
        lines.append(f"Title: {brief.title}")
    if brief.scene_description:
        lines.append(f"Scene to render: {brief.scene_description}")
    if brief.key_elements:
        lines.append("Key elements:")
        lines.extend(f"  - {item}" for item in brief.key_elements)
    if brief.relationships:
        lines.append("Relationships / flow:")
        lines.extend(f"  - {item}" for item in brief.relationships)
    if brief.labels_required:
        lines.append("Required labels: " + ", ".join(brief.labels_required))
    zones = brief.exploration_zones or build_fallback_exploration_zones(brief)
    if zones:
        lines.append(
            "Interactive exploration layout (MUST render on the image):"
        )
        if brief.interactive_title:
            lines.append(f"  Banner title on image: {brief.interactive_title}")
        for z in zones:
            region = f" at {z.region_hint}" if z.region_hint else ""
            lines.append(
                f"  - Zone {z.zone}: bold circled number '{z.zone}'{region}; "
                f"short label '{z.label}'; visually distinct highlight color per zone."
            )
    if brief.style_notes:
        lines.append(f"Style: {brief.style_notes}")
    if brief.avoid:
        lines.append("Avoid: " + "; ".join(brief.avoid))
    return "\n".join(lines)


def _brief_from_dict(parsed: Dict[str, Any], *, source: str) -> Optional[ImageGenerationBrief]:
    def _list(key: str) -> List[str]:
        value = parsed.get(key) or []
        if not isinstance(value, list):
            return None
        return [str(item).strip() for item in value if str(item).strip()]

    topic = str(parsed.get("topic") or "").strip()
    if not topic:
        return None

    return ImageGenerationBrief(
        topic=topic,
        diagram_type=str(
            parsed.get("diagram_type") or "realistic_educational_illustration"
        ).strip(),
        title=str(parsed.get("title") or topic[:80]).strip(),
        scene_description=str(parsed.get("scene_description") or "").strip(),
        key_elements=_list("key_elements") or [],
        relationships=_list("relationships") or [],
        labels_required=_list("labels_required") or [],
        exploration_zones=_parse_exploration_zones(parsed.get("exploration_zones")),
        interactive_title=str(parsed.get("interactive_title") or "").strip(),
        style_notes=str(parsed.get("style_notes") or "").strip(),
        avoid=_list("avoid") or [],
        source=source,
    )


def _fallback_brief(state: Dict[str, Any]) -> ImageGenerationBrief:
    from services.context_resolver import _last_substantive_user_topic, wants_flowchart

    query = str(state.get("query_en") or state.get("query") or "").strip()
    history = state.get("conversation_history") or []
    subjects = state.get("subjects") or []
    subject_hint = ", ".join(subjects[:3]) if isinstance(subjects, list) else ""
    session_topic = _session_topic(state.get("session_metadata") or {})
    tutor_context = _last_assistant_excerpt(history)
    is_flowchart = bool(state.get("wants_flowchart")) or wants_flowchart(query)

    topic = subject_hint or session_topic or query
    substantive = _last_substantive_user_topic(history, skip_text=query)
    if substantive and (len(query) < 160 or _VAGUE_REQUEST.search(query)):
        topic = substantive[:400]
    elif _VAGUE_REQUEST.search(query) and tutor_context:
        topic = tutor_context.split(".")[0][:240].strip() or topic

    key_elements: List[str] = []
    if tutor_context:
        key_elements.append("Use concepts from the ongoing tutor explanation.")
    if subject_hint:
        key_elements.append(f"Subject focus: {subject_hint}")

    if is_flowchart:
        scene = (
            f"A clear labeled process flowchart for: {topic}. "
            "Use standard flowchart symbols (start/end ovals, process rectangles, "
            "decision diamonds, arrows). Show the main steps in logical order with "
            "readable English labels on every box. School-appropriate for any subject."
        )
        if tutor_context:
            scene = f"{scene} Lesson context: {tutor_context[:400]}"
        diagram_type = "process_flowchart"
        style_notes = (
            "Clean 2D flowchart; high contrast; legible labels; no photorealistic rendering."
        )
        avoid = ["photorealistic", "3D render", "clipart faces", "watermarks"]
    else:
        scene = (
            f"A clear, detailed educational illustration of {topic}. "
            "Show the main concepts, parts, steps, or relationships with accurate callout labels. "
            "Choose the best visual form for the subject (labeled diagram, map, timeline, "
            "cross-section, experiment setup, historical scene, etc.). "
            "Not a bare icon or oversimplified clipart."
        )
        if tutor_context:
            scene = f"{scene} Context from lesson: {tutor_context[:400]}"
        diagram_type = "realistic_educational_illustration"
        style_notes = (
            "Textbook or classroom poster quality; clear labels; rich but accurate detail; "
            "all labels spelled correctly in English; style matched to the subject domain."
        )
        avoid = ["simple diagram", "clipart", "icon-only", "watermarks"]

    elements = key_elements or ["Main structures and processes for the topic"]
    brief = ImageGenerationBrief(
        topic=topic,
        diagram_type=diagram_type,
        title=topic[:80],
        scene_description=scene,
        key_elements=elements,
        relationships=[],
        labels_required=[],
        interactive_title=topic[:60] if topic else "Educational illustration",
        style_notes=style_notes,
        avoid=avoid,
        source="fallback",
    )
    brief.exploration_zones = build_fallback_exploration_zones(brief)
    return brief


def _fallback_plan(state: Dict[str, Any]) -> ImageGenerationPlan:
    brief = _fallback_brief(state)
    zones = brief.exploration_zones
    return ImageGenerationPlan(
        reasoning="Fallback planner: inferred illustration topic from chat context.",
        action="generate_diagram",
        student_response=enrich_student_response_for_exploration(
            "Here is a labeled illustration of what we were discussing. "
            "Ask about any numbered part if you want a deeper explanation.",
            zones,
        ),
        brief=brief,
        source="fallback",
    )


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    if "```" in text:
        text = text.split("```", 1)[-1]
        if text.startswith("json"):
            text = text[4:]
        text = text.split("```", 1)[0]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_plan_json(raw: str) -> Optional[ImageGenerationPlan]:
    parsed = _extract_json_object(raw)
    if not parsed:
        return None

    action = str(parsed.get("action") or "").strip().lower()
    if action not in {"generate_diagram", "respond_text_only", "ask_clarification"}:
        return None

    student_response = str(parsed.get("student_response") or "").strip()
    if not student_response:
        return None

    reasoning = str(parsed.get("reasoning") or "").strip()
    brief: Optional[ImageGenerationBrief] = None
    diagram = parsed.get("diagram")
    if isinstance(diagram, dict):
        brief = _brief_from_dict(diagram, source="llm")

    if action == "generate_diagram" and brief is None:
        return None

    if action != "generate_diagram":
        brief = None

    return ImageGenerationPlan(
        reasoning=reasoning or "Agent planned reaction from chat context.",
        action=action,  # type: ignore[arg-type]
        student_response=student_response,
        brief=brief,
        source="llm",
    )


class ImageGenerationPlanner:
    """Reason about chat context and decide how to react to image requests."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm

    async def plan(self, state: Dict[str, Any]) -> ImageGenerationPlan:
        fallback = _fallback_plan(state)
        if self._llm is None:
            return fallback

        query = str(state.get("query_en") or state.get("query") or "").strip()
        subjects = state.get("subjects") or []
        subject_hint = ", ".join(subjects[:3]) if isinstance(subjects, list) else "General"
        grade = str(state.get("student_grade") or "B")
        grade_style = _GRADE_STYLE.get(grade, _GRADE_STYLE["B"])
        from services.message_modality import format_history_with_modality

        history_block = format_history_with_modality(state.get("conversation_history"))
        session_topic = _session_topic(state.get("session_metadata") or "")
        ctx_block = str(state.get("context_resolution_block") or "").strip()
        wants_visual = bool(state.get("wants_visual_output"))
        wants_fc = bool(state.get("wants_flowchart"))

        prompt = (
            "You are VidyaAI, an educational tutor agent for all school and college subjects.\n"
            "Read the FULL conversation before the latest line—short replies like 'yes' or "
            "'flowchart from your end' refer to the ongoing topic in prior turns.\n"
            "Think step by step:\n"
            "1. What topic and subject is the student studying (from the whole thread, not only the last line)?\n"
            "2. Do they want a generated visual (flowchart vs illustration/diagram), text only, or clarification?\n"
            "3. Is there enough chat context to create a useful visual?\n"
            "4. What should you tell the student, and what visual spec (if any) should be generated?\n\n"
            "Rules:\n"
            "- Never pass vague student wording verbatim as the topic.\n"
            "- ALWAYS correct spelling/grammar of the topic, title, interactive_title, labels, "
            "and every on-image string. Use the correct subject term even if the "
            "student misspelled it (e.g. 'photosunthesis' -> 'Photosynthesis', 'mitocondria' -> "
            "'Mitochondria'). NEVER reproduce the student's typo on the image.\n"
            "- Resolve 'this/it/that' using prior tutor and student messages.\n"
            "- If the student explicitly asks to generate/draw/create an image or diagram "
            "(including 'generate me an image of …', 'draw a diagram of …'), you MUST use "
            "generate_diagram unless the topic is unsafe or completely non-educational.\n"
            "- Do NOT use respond_text_only or ask_clarification when they asked for an image "
            "of a teachable topic—build the diagram spec instead.\n"
            "- Use ask_clarification only when the subject is unsafe, non-educational, or "
            "impossible to illustrate for school.\n"
            "- Use respond_text_only only when they did NOT ask for a generated visual.\n"
            "- student_response must be warm, concise, and grade-appropriate (1-3 sentences).\n"
            "- student_response must NOT say the image is clickable, tappable, or interactive in the UI.\n"
            "- student_response must NOT say 'based on your request', 'here is a diagram I generated', "
            "or similar meta phrasing about fulfilling an image request.\n"
            "- student_response should teach in context (what the visual shows and why it matters); "
            "optionally invite text questions about labeled or numbered parts—never 'click' language.\n"
            "- When action is generate_diagram, fill diagram with topic, diagram_type, title, "
            "scene_description, key_elements, relationships, labels_required, exploration_zones, "
            "interactive_title, style_notes, avoid.\n"
            "- exploration_zones: 6–8 objects, each "
            '{ "zone": 1, "label": "...", "region_hint": "where on image", '
            '"curiosity_question": "open question about that part" }.\n'
            "- interactive_title: short banner text for the image only (e.g. key topic title)—not UI instructions.\n"
            "- diagram_type: choose what fits the subject—e.g. labeled_diagram, process_flowchart, "
            "timeline, map, anatomical_view, concept_map, historical_scene, experiment_setup, "
            "realistic_educational_illustration—not 'simple icon' or bare clipart.\n"
            "- scene_description: 4-6 sentences—art-director brief for THIS subject: layout, "
            "what to show, labeling style, notation (dates, formulas, units) if relevant, "
            "and what each region teaches.\n"
            "- labels_required: at least 8–10 exact English callout strings; correct spelling; "
            "use domain-appropriate notation (SI units, dates, names, symbols) when standard.\n"
            "- key_elements: at least 8–10 concrete parts, steps, regions, or concepts to depict "
            "(examples vary by subject: organelles, battle sites, graph axes, literary characters, "
            "machine parts, historical figures, geometric shapes, etc.).\n"
            "- relationships: at least 4–6 directed links (A -> B: cause/effect, flow, sequence, "
            "comparison, dependency) as appropriate to the topic.\n"
            "- style_notes: textbook or classroom poster quality; clear and accurate; "
            "NOT flat clipart, icon-only, or oversimplified blocks without labels.\n"
            "- avoid must include: clipart, icon-only, stick figures (unless topic requires), "
            "watermarks, oversimplified blocks without detail.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "reasoning": "your step-by-step thinking",\n'
            '  "action": "generate_diagram|respond_text_only|ask_clarification",\n'
            '  "student_response": "what you say to the student",\n'
            '  "diagram": null or { "topic": "...", "diagram_type": "...", "scene_description": "...", ... }\n'
            "}\n\n"
            f"{_EDUCATIONAL_AUDIENCE} Complexity: {grade_style}\n"
            f"Subject context: {subject_hint}\n"
            f"Session topic hint: {session_topic or 'none'}\n"
            f"Context resolver says visual_output={wants_visual} flowchart={wants_fc}\n"
            f"Current student request: {query}\n\n"
        )
        if ctx_block:
            prompt += f"Resolved session context:\n{ctx_block}\n\n"
        prompt += f"Recent conversation:\n{history_block or '(no prior turns)'}\n"
        prompt += GLOBAL_GUARDRAIL_RULES

        try:
            resp = await self._llm.ainvoke(
                prompt,
                config={"max_tokens": max(settings.query_analysis_tokens * 6, 500)},
            )
            usage = getattr(resp, "usage_metadata", None) or getattr(
                resp, "response_metadata", {}
            ).get("token_usage", {})
            i_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            o_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            model_name = getattr(self._llm, "model_name", settings.model_name)

            plan = _parse_plan_json(str(resp.content or ""))
            if plan is None:
                logger.warning("Image generation planner returned invalid JSON; using fallback.")
                fallback.input_tokens = i_tokens
                fallback.output_tokens = o_tokens
                fallback.model_name = model_name
                return fallback

            plan.input_tokens = i_tokens
            plan.output_tokens = o_tokens
            plan.model_name = model_name
            logger.info(
                "Image generation plan action=%s reasoning=%s",
                plan.action,
                (plan.reasoning or "")[:200],
            )
            return plan
        except Exception as exc:
            logger.warning("Image generation planning failed: %s", exc)
            return fallback


class ImagePromptComposer:
    """Legacy adapter that returns only the diagram prompt portion of a plan."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._planner = ImageGenerationPlanner(llm=llm)

    async def compose(self, state: Dict[str, Any]) -> ComposedImagePrompt:
        plan = await self._planner.plan(state)
        brief = plan.brief or _fallback_brief(state)
        return ComposedImagePrompt(
            brief=brief,
            prompt_text=format_brief_as_prompt(brief),
            input_tokens=plan.input_tokens,
            output_tokens=plan.output_tokens,
            model_name=plan.model_name,
        )


def college_grade_style(grade: str) -> str:
    return _GRADE_STYLE.get(str(grade or "B"), _GRADE_STYLE["B"])


__all__ = [
    "ImageExplorationZone",
    "ImageGenerationBrief",
    "ImageGenerationPlan",
    "ImageGenerationPlanner",
    "ComposedImagePrompt",
    "ImagePromptComposer",
    "build_fallback_exploration_zones",
    "enrich_student_response_for_exploration",
    "exploration_zones_to_dicts",
    "format_brief_as_prompt",
    "format_conversation_history",
    "college_grade_style",
]
