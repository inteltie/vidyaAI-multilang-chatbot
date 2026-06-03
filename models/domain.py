"""Pydantic models for API requests and responses."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv  
load_dotenv()

ALLOWED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class ImageInput(BaseModel):
    """Base64 image attachment for interactive chat."""

    mime_type: str = Field(..., description="image/jpeg, image/png, or image/webp")
    data: str = Field(..., description="Base64-encoded image bytes without data-URL prefix")

    @field_validator("mime_type")
    @classmethod
    def validate_mime(cls, v: str) -> str:
        normalized = (v or "").strip().lower()
        if normalized not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(
                f"Unsupported mime_type '{v}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}"
            )
        return normalized

    @field_validator("data")
    @classmethod
    def validate_data_non_empty(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("Image data must be non-empty")
        return v.strip()


class ImageExplorationZoneOutput(BaseModel):
    """Numbered region on a generated illustration for curiosity-driven follow-up in chat."""

    zone: int = Field(..., ge=1, description="Zone number shown on the image (1, 2, 3, …)")
    label: str = Field(..., description="Short name of the component or region")
    region_hint: str = Field(
        "",
        description="Where on the image this zone appears (helps UI highlight)",
    )
    curiosity_question: str = Field(
        "",
        description="Open question inviting the student to explore this zone",
    )


class GeneratedImageOutput(BaseModel):
    """Generated diagram metadata returned to the client (URL only, no base64)."""

    image_id: str = ""
    url: str = ""
    mime_type: str = "image/png"
    prompt_used: str = ""
    created_at: str = ""
    interactive_title: str = Field(
        "",
        description="Engaging banner text rendered on or paired with the illustration",
    )
    exploration_zones: List[ImageExplorationZoneOutput] = Field(
        default_factory=list,
        description="Numbered exploration zones; student can reply with a zone id in chat",
    )


class MessageModalityTurn(BaseModel):
    """Per-message flags for image input vs generated diagram output."""

    role: str
    content_preview: str = ""
    has_image_input: bool = False
    image_input_ids: List[str] = Field(default_factory=list)
    has_generated_image: bool = False
    generated_image_ids: List[str] = Field(default_factory=list)
    generated_image_urls: List[str] = Field(default_factory=list)


class GeneratedImageMetadata(BaseModel):
    """Generated diagram metadata for session history (no base64)."""

    image_id: str
    url: str
    mime_type: str = "image/png"
    prompt_used: str = ""
    created_at: str = ""


class SessionGeneratedImagesResponse(BaseModel):
    """List of generated diagrams for a chat session."""

    user_session_id: str
    generated_images: List[GeneratedImageMetadata] = Field(default_factory=list)


class SessionMessageRecord(BaseModel):
    """Chat message with image input/output flags."""

    role: str
    text: str
    timestamp: str = ""
    has_image_input: bool = False
    image_input_ids: List[str] = Field(default_factory=list)
    has_generated_image: bool = False
    generated_image_ids: List[str] = Field(default_factory=list)
    generated_image_urls: List[str] = Field(default_factory=list)


class SessionMessagesResponse(BaseModel):
    """Full message history for a session with per-message image flags."""

    user_session_id: str
    messages: List[SessionMessageRecord] = Field(default_factory=list)


class QueryIntent(str, Enum):
    """Supported query intents for the educational chatbot."""

    CONCEPT_EXPLANATION = "concept_explanation"
    HOMEWORK_HELP = "homework_help"
    EXAM_PREP = "exam_prep"
    DOUBT_RESOLUTION = "doubt_resolution"
    OFF_TOPIC = "off_topic"


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    
    user_session_id: str = Field(..., description="Session identifier for conversation continuity")
    user_id: str = Field(..., description="Unique identifier for the user")
    user_type: str = Field(..., pattern="^(student|teacher)$", description="User type: 'student' or 'teacher'")
    query: str = Field(..., description="User's message or question")
    language: str = Field("en", description="ISO code of user's language, default 'en'")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for vector DB")
    agent_mode: Optional[str] = Field(
        "standard",
        description="Agent mode: 'standard' (default StudentAgent) or 'interactive' (step-by-step InteractiveStudentAgent)"
    )
    student_grade: Optional[Literal["A", "B", "C", "D"]] = Field(
        "B",
        description="Student grade: 'A' (Top), 'B' (Average), 'C' (Below Average), 'D' (Foundational)"
    )
    images: Optional[List[ImageInput]] = Field(
        default_factory=list,
        description="Optional base64 images (interactive mode only, max 3)",
    )

    @field_validator("images", mode="before")
    @classmethod
    def normalize_images(cls, v: Any) -> List[Any]:
        """Accept missing/null images and normalize to empty list."""
        if v is None:
            return []
        return v

    @model_validator(mode="after")
    def validate_images(self) -> "ChatRequest":
        imgs = self.images or []
        max_images = int(os.getenv("MAX_IMAGES_PER_REQUEST") or 3)
        if len(imgs) > max_images:
            raise ValueError(f"At most {max_images} images allowed per request")
        if not imgs:
            return self
        if not (self.query or "").strip():
            raise ValueError("query is required when images are provided")
        return self
    
    @field_validator('filters', mode='before')
    @classmethod
    def parse_filters(cls, v):
        """Convert string filters to dict, handle empty cases."""
        if v is None:
            return None
        if isinstance(v, str):
            # Handle empty string or string "{}"
            if not v or v.strip() in ['{}', '']:
                return None
            # Try to parse JSON string
            import json
            try:
                parsed = json.loads(v)
                return parsed if parsed else None
            except json.JSONDecodeError:
                return None
        if isinstance(v, dict):
            # Return None for empty dict
            return v if v else None
        return v


class ChatResponse(BaseModel):
    """Chatbot response payload."""

    user_session_id: str
    message: str
    intent: str
    language: str
    citations: List["DocumentCitation"] = Field(
        default_factory=list,
        description="Citations for documents used during retrieval.",
    )
    llm_calls: int = Field(
        0,
        description="Total number of LLM calls used to answer this request.",
    )
    input_tokens: int = Field(
        0,
        description="Total input tokens used for this request cycle.",
    )
    output_tokens: int = Field(
        0,
        description="Total output tokens used for this request cycle.",
    )
    total_tokens: int = Field(
        0,
        description="Total tokens (input + output) used for this request cycle.",
    )
    background_input_tokens: int = Field(
        0,
        description="Input tokens used by background summarization since last response.",
    )
    background_output_tokens: int = Field(
        0,
        description="Output tokens used by background summarization since last response.",
    )
    background_total_tokens: int = Field(
        0,
        description="Total tokens used by background summarization since last response.",
    )
    total_tokens_with_background: int = Field(
        0,
        description="Total tokens including background summarization since last response.",
    )
    # Granular token tracking for gpt-4.1-nano vs gpt-4o-mini
    primary_input_tokens: int = Field(0, description="Input tokens for main Reasoning model (Mini)")
    primary_output_tokens: int = Field(0, description="Output tokens for main Reasoning model (Mini)")
    aux_input_tokens: int = Field(0, description="Input tokens for auxiliary models (Nano)")
    aux_output_tokens: int = Field(0, description="Output tokens for auxiliary models (Nano)")
    embedding_tokens: int = Field(0, description="Estimated embedding tokens used for retrieval in this request.")
    token_usage_by_model: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Per-model input/output token usage for this request.",
    )
    cost_by_model_usd: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-model USD cost breakdown for this request.",
    )
    cost_usd_llm: float = Field(0.0, description="Total USD cost for foreground LLM calls in this request.")
    cost_usd_web_search: float = Field(
        0.0, description="USD cost for web search tool calls and billed search tokens."
    )
    cost_usd_vision: float = Field(0.0, description="USD cost for image/vision analysis in this request.")
    cost_usd_background: float = Field(0.0, description="USD cost from background summarization tokens.")
    cost_usd_embedding: float = Field(0.0, description="USD embedding cost for this request.")
    cost_usd_total: float = Field(
        0.0,
        description="Total request cost in USD (LLM + web search + vision + background + embedding).",
    )
    cost_by_component_usd: Dict[str, float] = Field(
        default_factory=dict,
        description="USD cost grouped by component (chat, web_search, vision, etc.).",
    )
    web_search_calls: int = Field(0, description="Billable web_search_preview tool invocations this request.")
    images_analyzed: int = Field(0, description="Images sent to vision analysis this request.")
    generated_images: List[GeneratedImageOutput] = Field(
        default_factory=list,
        description="Generated diagram metadata for this request (URL only; fetch via url).",
    )
    image_generation_calls: int = Field(
        0, description="Billable image_generation tool invocations this request."
    )
    cost_usd_image_generation: float = Field(
        0.0, description="USD cost for image_generation tool calls this request."
    )
    image_generation_backend: str = Field(
        "",
        description="Image generation provider (google).",
    )
    image_model_used: str = Field(
        "",
        description="Model used for generation (e.g. gemini-2.5-flash-image).",
    )
    image_generation_status: str = Field(
        "",
        description="Generation status: success, blocked, disabled, mode_restricted, provider_error, or no_image_returned.",
    )
    image_generation_error_reason: str = Field(
        "",
        description="Machine-readable reason when generation did not produce images.",
    )
    orchestration_route: str = Field("", description="Chosen orchestration route for this turn.")
    route_reason: str = Field("", description="Human-readable reason for route selection.")
    route_confidence: float = Field(0.0, description="Router confidence score [0..1].")
    route_fallback_reason: str = Field("", description="Fallback or guard reason for routing.")
    web_rewrite_enabled: bool = Field(False, description="Whether web query rewrite was enabled for this request.")
    web_query_original: str = Field("", description="Original query considered for web fallback.")
    web_query_used: str = Field("", description="Final query sent to web search (rewritten or original).")
    rewrite_input_tokens: int = Field(0, description="Input tokens used by web query rewriter.")
    rewrite_output_tokens: int = Field(0, description="Output tokens used by web query rewriter.")
    rewrite_guard_reason: str = Field("", description="Reason reported by rewrite intent guard.")
    rewrite_overlap_ratio: float = Field(0.0, description="Token overlap ratio between original and rewritten query.")
    rewrite_used_original_due_to_guard: bool = Field(
        False,
        description="Whether guard forced fallback to original query due to detected rewrite drift.",
    )
    user_message_had_image_input: bool = Field(
        False,
        description="True when this request's user message included uploaded image(s).",
    )
    assistant_message_has_generated_image: bool = Field(
        False,
        description="True when this request's assistant reply included a generated diagram.",
    )
    message_modality_timeline: List[MessageModalityTurn] = Field(
        default_factory=list,
        description="Recent turns with flags for image upload vs diagram generation.",
    )
    memory_degraded: bool = Field(
        False,
        description="True when chat history loaded from Redis only (MongoDB unreachable).",
    )


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str


class DocumentCitation(BaseModel):
    """Citation for a retrieved document."""

    id: str
    score: float
    subject_id: Optional[int] = None
    topics: Optional[str] = None
    class_id: Optional[int] = None
    teacher_id: Optional[int] = None
    lecture_id: Optional[str] = None
    transcript_id: Optional[str] = None
    chunk_id: Optional[str] = None



