"""Pydantic models for API requests and responses."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv  
load_dotenv()

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





