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


def merge_token_usage_by_model(
    left: Dict[str, Dict[str, int]],
    right: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Merge model-token usage maps by summing input/output counters."""
    merged: Dict[str, Dict[str, int]] = {}
    for source in (left or {}, right or {}):
        for model_name, usage in (source or {}).items():
            if not model_name:
                continue
            current = merged.setdefault(model_name, {"input": 0, "output": 0})
            current["input"] += int((usage or {}).get("input", 0) or 0)
            current["output"] += int((usage or {}).get("output", 0) or 0)
    return merged


def merge_usage_by_component(
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge per-component usage maps (chat, web_search, vision, image_gate)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for source in (left or {}, right or {}):
        for component, comp_data in (source or {}).items():
            if not component:
                continue
            target = merged.setdefault(component, {})
            for key, val in (comp_data or {}).items():
                if key in ("calls", "images"):
                    target[key] = int(target.get(key, 0) or 0) + int(val or 0)
                elif key == "fee_usd":
                    target[key] = float(target.get(key, 0) or 0) + float(val or 0)
                elif key in ("backend", "model") and val:
                    target[key] = val
                elif isinstance(val, dict):
                    row = target.setdefault(key, {"input": 0, "output": 0})
                    row["input"] += int(val.get("input", 0) or 0)
                    row["output"] += int(val.get("output", 0) or 0)
    return merged


class Document(TypedDict, total=False):
    """Representation of a retrieved document from Pinecone."""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class ConversationTurn(TypedDict, total=False):
    """Single conversation turn stored in Redis memory."""

    role: Literal["user", "assistant"]
    content: str
    has_image_input: bool
    image_input_ids: List[str]
    image_input_urls: List[str]
    has_generated_image: bool
    generated_image_ids: List[str]
    generated_image_urls: List[str]


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


class AgentFinding(TypedDict, total=False):
    """Structured finding published by a specialist agent/node."""

    source: str
    kind: str
    summary: str
    payload: Dict[str, Any]
    priority: int


class UploadedImageRecord(TypedDict, total=False):
    """Persisted description of an uploaded image (no raw bytes)."""

    image_id: str
    upload_id: str
    user_query: str
    url: str
    descriptions: List[str]
    summary_line: str
    image_context: str
    uploaded_at: str


class GeneratedImageRecord(TypedDict, total=False):
    """Persisted metadata for a generated diagram (S3 URL)."""

    image_id: str
    url: str
    mime_type: str
    prompt_used: str
    created_at: str


def merge_findings(
    left: List[AgentFinding],
    right: List[AgentFinding],
) -> List[AgentFinding]:
    """Append findings from parallel or sequential node updates."""
    return list(left or []) + list(right or [])


class AgentState(TypedDict, total=False):
    """Full state passed between LangGraph nodes."""

    # Input fields (from API request)
    user_session_id: str
    user_id: str
    user_type: Literal["student", "teacher"]
    query: str
    images: List[Dict[str, Any]]
    has_images: bool
    image_modality: Literal["none", "analyze", "generate"]
    orchestration_route: Literal[
        "none", "retriever", "web", "hybrid", "image_generate", "image_analyze"
    ]
    route_reason: str
    route_confidence: float
    route_fallback_reason: str
    generated_images: List[Dict[str, Any]]
    remembered_generated_images: List[GeneratedImageRecord]
    message_modality_timeline: List[Dict[str, Any]]
    new_generated_image_records: List[GeneratedImageRecord]
    remembered_images: List[UploadedImageRecord]
    image_descriptions: List[str]
    image_context: str
    effective_query: str
    context_resolution_block: str
    is_context_resolved_followup: bool
    wants_visual_output: bool
    wants_flowchart: bool
    response_strategy: str
    preferred_response_format: str
    query_understanding: Dict[str, Any]
    # Authoritative, context-aware plan for the current turn (produced by
    # OrchestratorNode; consumed by analyze_query + route_decision).
    turn_plan: Dict[str, Any]
    image_inject_level: Literal["none", "compact", "full"]
    image_inject_block: str
    image_gate_reason: str
    new_uploaded_image_records: List[UploadedImageRecord]
    interactive_turn_type: str
    agent_findings: Annotated[List[AgentFinding], merge_findings]
    translation_action: Literal["skip", "translate"]
    translation_skip_reason: str
    language: str
    preferred_response_language: str
    language_preference_locked: bool
    agent_mode: str
    student_grade: Literal["A", "B", "C", "D"]
    subjects: Annotated[List[str], merge_list]
    prefilled_observations: List[Dict[str, Any]]
    rag_quality: Optional[Literal["low", "medium", "high"]]
    web_fallback_reason: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    is_correction: bool
    clarification_message: Optional[str]
    memory_intent: Optional[Literal["name", "class_level"]]
    is_time_sensitive_query: bool
    is_current_affairs_query: bool

    # Session context (extracted from history + current query)
    session_metadata: Annotated[SessionMetadata, merge_metadata]
    request_filters: Annotated[Dict[str, Any], merge_metadata]
    extracted_metadata: Annotated[SessionMetadata, merge_metadata]

    # Processing fields
    query_en: str
    detected_language: str
    detected_language_confidence: float
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
    # Token tracking by model type (Nano vs Mini)
    primary_input_tokens: Annotated[int, operator.add]
    primary_output_tokens: Annotated[int, operator.add]
    aux_input_tokens: Annotated[int, operator.add]
    aux_output_tokens: Annotated[int, operator.add]
    token_usage_by_model: Annotated[Dict[str, Dict[str, int]], merge_token_usage_by_model]
    usage_by_component: Annotated[Dict[str, Dict[str, Any]], merge_usage_by_component]
    web_search_calls: Annotated[int, operator.add]
    images_analyzed: Annotated[int, operator.add]
    image_generation_calls: Annotated[int, operator.add]
    image_generation_backend: str
    image_model_used: str
    image_generation_status: str
    image_generation_error_reason: str
    embedding_tokens: Annotated[int, operator.add]
    web_rewrite_enabled: bool
    web_query_original: str
    web_query_used: str
    rewrite_input_tokens: Annotated[int, operator.add]
    rewrite_output_tokens: Annotated[int, operator.add]
    rewrite_guard_reason: str
    rewrite_overlap_ratio: float
    rewrite_used_original_due_to_guard: bool

    # Legacy / Aggregate fields (keep for compatibility/simplicity)
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


