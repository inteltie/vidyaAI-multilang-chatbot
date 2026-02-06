"""Shared agent state definition for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, Annotated
import operator
from langchain_core.messages import BaseMessage
from models import QueryIntent


def merge_timings(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Merge two timing dictionaries."""
    return {**(left or {}), **(right or {})}


def merge_citations(left: List[Citation], right: List[Citation]) -> List[Citation]:
    """Merge citations ensuring uniqueness by document ID."""
    if not left: return right or []
    if not right: return left or []
    
    # Combined list
    combined = (left or []) + (right or [])
    
    # Ensure uniqueness by ID
    seen_ids = set()
    unique_citations = []
    for citation in combined:
        cite_id = citation.get("id")
        if cite_id not in seen_ids:
            unique_citations.append(citation)
            seen_ids.add(cite_id)
            
    return unique_citations


def merge_metadata(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two metadata dictionaries."""
    return {**(left or {}), **(right or {})}


class Document(TypedDict, total=False):
    """Representation of a retrieved document from Pinecone."""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class ConversationTurn(TypedDict):
    """Single conversation turn stored in Redis memory."""

    role: Literal["user", "assistant"]
    content: str


class Citation(TypedDict, total=False):
    """Lightweight citation for a retrieved document."""

    id: str
    score: float
    title: Optional[str]
    subject: Optional[str]
    chapter: Optional[str]
    # Detail level for response (affects max_tokens)
    detail_level: Optional[Literal["brief", "default", "detailed"]]
    lecture_id: Optional[str]
    transcript_id: Optional[str]


class SessionMetadata(TypedDict, total=False):
    """Metadata extracted from conversation history or current query."""

    class_level: Optional[str]
    subject: Optional[str]
    chapter: Optional[str]
    lecture_id: Optional[str]  # NEW: lecture session ID if user mentions it
    last_topic: Optional[str]  # Brief summary of what was being discussed


def merge_list(left: List[str], right: List[str]) -> List[str]:
    """Combine two lists of strings, removing duplicates."""
    return list(set((left or []) + (right or [])))


class AgentState(TypedDict, total=False):
    """Full state passed between LangGraph nodes."""

    # Input fields (from API request)
    user_session_id: str
    user_id: str
    user_type: Literal["student", "teacher"]
    query: str
    language: str
    agent_mode: str
    student_grade: Literal["A", "B", "C", "D"]
    subjects: Annotated[List[str], merge_list]
    prefilled_observations: List[Dict[str, Any]]
    rag_quality: Optional[Literal["low", "medium", "high"]]
    validation_results: Optional[Dict[str, Any]]
    is_correction: bool
    clarification_message: Optional[str]

    # Session context (extracted from history + current query)
    session_metadata: Annotated[SessionMetadata, merge_metadata]
    request_filters: Annotated[Dict[str, Any], merge_metadata]
    extracted_metadata: Annotated[SessionMetadata, merge_metadata]

    # Processing fields
    query_en: str
    detected_language: str
    intent: QueryIntent
    is_context_reply: bool
    is_topic_shift: bool
    is_acknowledgment: bool
    query_type: str  # NEW: "acknowledgment" | "general_knowledge" | "curriculum_specific"
    documents: List[Document]
    conversation_history: List[Union[ConversationTurn, BaseMessage]]
    needs_context: bool
    citations: Annotated[List[Citation], merge_citations]
    timings: Annotated[Dict[str, float], merge_timings]
    llm_calls: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    is_session_restart: bool

    # Context gathering tracking
    context_gathering_attempts: int
    asked_for_fields: List[str]

    # Output fields
    response: str
    final_language: str
    is_translated: bool


