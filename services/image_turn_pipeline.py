"""Shared image analyze → gate → tutor-context prep for educational agents."""

from __future__ import annotations

from typing import Any, Dict

from agents.image_analyzer_agent import ImageAnalyzerAgent
from agents.image_context_gate_agent import ImageContextGateAgent
from services.tutor_context_composer import TutorContextComposer


async def prepare_image_turn(
    state: Dict[str, Any],
    *,
    image_analyzer: ImageAnalyzerAgent,
    image_gate: ImageContextGateAgent,
    tutor_composer: TutorContextComposer,
    turn_type: str = "opening",
) -> Dict[str, Any]:
    """Run vision analysis (if needed), gate injection, and tutor context assembly."""
    prep: Dict[str, Any] = {"interactive_turn_type": turn_type}
    working = dict(state)
    findings = list(state.get("agent_findings") or [])

    if state.get("images"):
        analyzer_updates = await image_analyzer(working)
        prep.update(analyzer_updates)
        working.update(analyzer_updates)
        findings = list(prep.get("agent_findings") or findings)

    working["agent_findings"] = findings
    working["interactive_turn_type"] = turn_type
    gate_updates = await image_gate(working)
    prep.update(gate_updates)
    working.update(gate_updates)
    findings = list(prep.get("agent_findings") or findings)
    working["agent_findings"] = findings

    tutor_input = tutor_composer.compose(working)
    prep["tutor_input"] = tutor_input
    prep["agent_findings"] = findings
    if working.get("effective_query"):
        prep["effective_query"] = working["effective_query"]
    return prep


__all__ = ["prepare_image_turn"]
