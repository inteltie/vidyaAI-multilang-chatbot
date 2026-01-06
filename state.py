"""Shared agent state definition for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from langchain_core.messages import BaseMessage
from models import QueryIntent


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
    transcript_id: Optional[int]


class SessionMetadata(TypedDict, total=False):
    """Metadata extracted from conversation history or current query."""

    class_level: Optional[str]
    subject: Optional[str]
    chapter: Optional[str]
    lecture_id: Optional[str]  # NEW: lecture session ID if user mentions it
    last_topic: Optional[str]  # Brief summary of what was being discussed


class AgentState(TypedDict, total=False):
    """Full state passed between LangGraph nodes."""

    # Input fields (from API request)
    user_session_id: str  # User session identifier (not lecture session)
    user_id: str
    user_type: Literal["student", "teacher"]
    query: str
    language: str
    agent_mode: str  # "standard" or "interactive"
    subjects: List[str]  # Detected subjects from query analysis
    prefilled_observations: List[Dict[str, Any]]  # Simulated tool results to skip first iteration
    rag_quality: Optional[Literal["low", "medium", "high"]]  # Quality of retrieved RAG documents
    validation_results: Optional[Dict[str, Any]]  # Results from ResponseValidator
    is_correction: bool  # Flag indicating if this is a corrected response turn
    clarification_message: Optional[str]  # Message to ask user for clarification

    # Session context (extracted from history + current query)
    session_metadata: SessionMetadata
    request_filters: Dict[str, Any]  # Filters provided in the API request
    extracted_metadata: SessionMetadata  # Metadata extracted from current query

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
    citations: List[Citation]
    timings: Dict[str, float]
    # exactly there is not a part but a topic i need to discuss
    llm_calls: int

    # Context gathering tracking
    context_gathering_attempts: int  # How many times we've asked for context
    asked_for_fields: List[str]  # Which fields we've asked for (to avoid repetition)

    # Output fields
    response: str
    final_language: str
    is_translated: bool  # NEW: Flag to indicate if response is already in target language


