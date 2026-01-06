"""LangGraph workflow definition for the educational chatbot."""

from __future__ import annotations

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
)
from models import QueryIntent
from state import AgentState


RouteKey = Literal["off_topic", "needs_context", "ok"]
AgentRouteKey = Literal["conversational", "educational"]


class ChatbotGraphBuilder:
    """Builder for the multi-agent LangGraph workflow with student/teacher routing."""

    def __init__(
        self,
        load_memory: LoadMemoryNode,
        analyze_query: AnalyzeQueryNode,
        conversational_agent: ConversationalAgentNode,
        student_agent: StudentAgentNode,
        interactive_student_agent: InteractiveStudentAgentNode,
        teacher_agent: TeacherAgentNode,
        groundedness_check: GroundednessCheckNode,
        translate_response: TranslateResponseNode,
        save_memory: SaveMemoryNode,
    ) -> None:
        self._load_memory = load_memory
        self._analyze_query = analyze_query
        self._conversational_agent = conversational_agent
        self._student_agent = student_agent
        self._interactive_student_agent = interactive_student_agent
        self._teacher_agent = teacher_agent
        self._groundedness_check = groundedness_check
        self._translate_response = translate_response
        self._save_memory = save_memory

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
        query_type = state.get("query_type", "curriculum_specific")
        
        if query_type == "conversational":
            return "conversational"
        else:  # curriculum_specific → route by user_type
            return "educational"

    @staticmethod
    def _route_after_context(state: AgentState) -> RouteKey:
        """Routing logic after `check_context` node."""
        intent = state.get("intent")
        needs_context = bool(state.get("needs_context"))

        if intent == QueryIntent.OFF_TOPIC:
            return "off_topic"
        if needs_context:
            return "needs_context"
        return "ok"

    @staticmethod
    def _route_after_validation(state: AgentState) -> Literal["pass", "fail"]:
        """Route based on validation results."""
        val_results = state.get("validation_results")
        if not val_results:
            return "pass"
            
        # If valid OR needs clarification (HITL), we pass through to translation
        if val_results.get("is_valid") or val_results.get("needs_clarification"):
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
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("conversational_agent", self._conversational_agent)
        graph.add_node("student_agent", self._student_agent)
        graph.add_node("interactive_student_agent", self._interactive_student_agent)
        graph.add_node("teacher_agent", self._teacher_agent)
        graph.add_node("groundedness_check", self._groundedness_check)
        graph.add_node("translate_response", self._translate_response)
        graph.add_node("save_memory", self._save_memory)

        # Linear flow to query analysis
        # Optimized Order: load -> analyze_query (now includes context parsing)
        graph.add_edge("load_memory", "analyze_query")

        # Route to appropriate agent
        # Routing now happens directly after 'analyze_query'
        graph.add_conditional_edges(
            "analyze_query",
            self._route_to_agent,
            {
                "conversational": "conversational_agent",
                "educational": "route_educational_user",
            },
        )
        
        # Add intermediate routing node for educational users
        graph.add_node("route_educational_user", lambda state: state)  # Pass-through node
        graph.add_conditional_edges(
            "route_educational_user",
            self._route_educational_user,
            {
                "student": "student_agent",
                "interactive": "interactive_student_agent",
                "teacher": "teacher_agent",
            },
        )

        # All agents go to translation (except educational which go to validation)
        graph.add_edge("conversational_agent", "translate_response")
        
        # Educational agents go to validation
        graph.add_edge("student_agent", "groundedness_check")
        graph.add_edge("interactive_student_agent", "groundedness_check")
        graph.add_edge("teacher_agent", "groundedness_check")

        # Validation routing
        graph.add_conditional_edges(
            "groundedness_check",
            self._route_after_validation,
            {
                "pass": "translate_response",
                "fail": "route_educational_user",  # Self-correction loop
            }
        )

        # Translation → Save → END
        graph.add_edge("translate_response", "save_memory")

        # Set entry point
        graph.set_entry_point("load_memory")
        graph.set_finish_point("save_memory")

        return graph

    def compile(self):
        """Compile and return the runnable graph."""
        graph = self.build()
        return graph.compile()


__all__ = ["ChatbotGraphBuilder"]


