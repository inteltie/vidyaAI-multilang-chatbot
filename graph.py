"""LangGraph workflow definition for the educational chatbot."""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from nodes import (
    ConversationalAgentNode,
    InteractiveStudentAgentNode,
    LoadMemoryNode,
    ParseSessionContextNode,
    SaveMemoryNode,
    StudentAgentNode,
    TeacherAgentNode,
    TranslateResponseNode,
    GroundednessCheckNode,
    RetrieveDocumentsNode,
    AnalyzeQueryNode,
    OrchestratorNode,
    RouteDecisionNode,
)
from models import QueryIntent
from state import AgentState

logger = logging.getLogger(__name__)

RouteKey = Literal["off_topic", "needs_context", "ok"]
AgentRouteKey = Literal["conversational", "educational", "image_generation"]


class ChatbotGraphBuilder:
    """Builder for the multi-agent LangGraph workflow with student/teacher routing."""

    def __init__(
        self,
        load_memory: LoadMemoryNode,
        orchestrate_turn: OrchestratorNode,
        analyze_query: AnalyzeQueryNode,
        conversational_agent: ConversationalAgentNode,
        student_agent: StudentAgentNode,
        interactive_student_agent: InteractiveStudentAgentNode,
        teacher_agent: TeacherAgentNode,
        retrieve_documents: RetrieveDocumentsNode,
        route_decision: RouteDecisionNode,
        groundedness_check: GroundednessCheckNode,
        translate_response: TranslateResponseNode,
        save_memory: SaveMemoryNode,
        image_generation_agent=None,
    ) -> None:
        self._load_memory = load_memory
        self._orchestrate_turn = orchestrate_turn
        self._analyze_query = analyze_query
        self._conversational_agent = conversational_agent
        self._student_agent = student_agent
        self._interactive_student_agent = interactive_student_agent
        self._teacher_agent = teacher_agent
        self._retrieve_documents = retrieve_documents
        self._route_decision = route_decision
        self._groundedness_check = groundedness_check
        self._translate_response = translate_response
        self._save_memory = save_memory
        self._image_generation_agent = image_generation_agent

    @staticmethod
    def _route_educational_user(state: AgentState) -> Literal["student", "interactive", "teacher"]:
        """Route to appropriate educational agent based on user_type and agent_mode."""
        user_type = state.get("user_type", "student")
        agent_mode = state.get("agent_mode", "standard")
        
        if user_type == "teacher":
            return "teacher"
        elif agent_mode == "interactive":
            return "interactive"
        else:
            return "student"  # Default to standard student agent

    @staticmethod
    def _route_to_agent(state: AgentState) -> AgentRouteKey:
        """Route to appropriate agent based on query_type."""
        route = state.get("orchestration_route")
        if route == "none":
            return "conversational"
        if route == "image_analyze":
            logger.info("Image analysis route selected; routing to educational pipeline.")
            return "educational"
        if route == "image_generate" or state.get("query_type") == "image_generation":
            logger.info("Image generation modality; routing to image_generation_agent.")
            return "image_generation"

        if route == "image_analyze" or state.get("has_images") or state.get("images"):
            logger.info(
                "Image upload present; routing to educational flow (vision via interactive agent)."
            )
            return "educational"

        query_type = state.get("query_type", "curriculum_specific")

        if query_type == "conversational":
            return "conversational"
        return "educational"

    @staticmethod
    def _route_after_validation(state: AgentState) -> Literal["pass", "fail"]:
        """Route based on validation results."""
        val_results = state.get("validation_results")
        if not val_results:
            return "pass"
            
        # If valid, we pass through to translation
        if val_results.get("is_valid"):
            return "pass"
        
        # Limit retries to 1 (is_correction is set by agent on retry)
        if state.get("is_correction"):
            logger.warning("Validation failed again on correction turn. Passing through to avoid infinite loops.")
            return "pass"
            
        return "fail"

    def build(self) -> StateGraph[AgentState]:
        """Construct the uncompiled StateGraph."""
        graph: StateGraph[AgentState] = StateGraph(AgentState)

        # Register nodes
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("orchestrate_turn", self._orchestrate_turn)
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("route_decision", self._route_decision)
        graph.add_node("conversational_agent", self._conversational_agent)
        if self._image_generation_agent is not None:
            graph.add_node("image_generation_agent", self._image_generation_agent)
        graph.add_node("student_agent", self._student_agent)
        graph.add_node("interactive_student_agent", self._interactive_student_agent)
        graph.add_node("teacher_agent", self._teacher_agent)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("groundedness_check", self._groundedness_check)
        graph.add_node("translate_response", self._translate_response)
        graph.add_node("save_memory", self._save_memory)

        # 1. Start with memory loading, then context-aware turn planning.
        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "orchestrate_turn")
        graph.add_edge("orchestrate_turn", "analyze_query")
        graph.add_edge("analyze_query", "route_decision")

        # 2. Main Routing: Conversational vs Educational
        route_map = {
            "conversational": "conversational_agent",
            "educational": "prepare_educational_flow",
        }
        if self._image_generation_agent is not None:
            route_map["image_generation"] = "image_generation_agent"
        graph.add_conditional_edges(
            "route_decision",
            self._route_to_agent,
            route_map,
        )

        # 3. Conversational Pipeline: Direct to translation
        graph.add_edge("conversational_agent", "translate_response")
        if self._image_generation_agent is not None:
            graph.add_edge("image_generation_agent", "translate_response")

        # 4. Educational Pipeline: Strictly Sequential (Retrieval -> Agent -> Validation)
        # This prevents the race condition and duplication
        graph.add_node("prepare_educational_flow", lambda state: {})
        graph.add_edge("prepare_educational_flow", "retrieve_documents")
        
        # After retrieval, route to specific educational agent
        graph.add_edge("retrieve_documents", "route_educational_user")
        
        graph.add_node("route_educational_user", lambda state: {}) # Pass-through
        graph.add_conditional_edges(
            "route_educational_user",
            self._route_educational_user,
            {
                "student": "student_agent",
                "interactive": "interactive_student_agent",
                "teacher": "teacher_agent",
            },
        )

        # 5. Sequential Validation and Translation for Educational Flow
        # This ensures the groundedness check validates the FINAL English response BEFORE translation.
        
        # Link educational agents to groundedness check
        graph.add_edge("student_agent", "groundedness_check")
        graph.add_edge("interactive_student_agent", "groundedness_check")
        graph.add_edge("teacher_agent", "groundedness_check")
        
        # Routing after validation completes
        graph.add_conditional_edges(
            "groundedness_check",
            self._route_after_validation,
            {
                "pass": "translate_response_educational",
                "fail": "route_educational_user",  # Self-correction loop
            }
        )

        graph.add_node("translate_response_educational", self._translate_response)
        graph.add_edge("translate_response_educational", "save_memory")

        # 6. Finalization: Save -> END
        graph.add_edge("translate_response", "save_memory")
        graph.add_edge("save_memory", END)
        graph.set_finish_point("save_memory")

        return graph

    def compile(self):
        """Compile and return the runnable graph."""
        graph = self.build()
        return graph.compile()


__all__ = ["ChatbotGraphBuilder"]
