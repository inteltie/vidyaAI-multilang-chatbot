from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
from beanie import Document
from pydantic import BaseModel, Field, field_validator
from pymongo import IndexModel, ASCENDING

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current time in IST."""
    return datetime.now(IST)

class UploadedImageRecord(BaseModel):
    """Persisted analyzer output for an uploaded image (no raw bytes)."""

    image_id: str
    upload_id: str
    user_query: str
    descriptions: List[str] = Field(default_factory=list)
    summary_line: str = ""
    image_context: str = ""
    uploaded_at: datetime = Field(default_factory=get_ist_now)


class GeneratedImageRecord(BaseModel):
    """Persisted metadata for a generated diagram (bytes on disk, not in Mongo)."""

    image_id: str
    url: str
    mime_type: str = "image/png"
    prompt_used: str = ""
    exploration_zones: List[Dict[str, Any]] = Field(default_factory=list)
    interactive_title: str = ""
    created_at: datetime = Field(default_factory=get_ist_now)


class ChatMessage(BaseModel):
    """Embedded model for a single chat message."""
    role: str
    text: str
    timestamp: datetime = Field(default_factory=get_ist_now)
    has_image_input: bool = Field(
        False, description="True when the user attached image(s) on this turn."
    )
    image_input_ids: List[str] = Field(
        default_factory=list,
        description="Upload/image ids for user-provided images on this turn.",
    )
    has_generated_image: bool = Field(
        False, description="True when the assistant reply included a generated diagram."
    )
    generated_image_ids: List[str] = Field(
        default_factory=list,
        description="Generated diagram ids produced on this assistant turn.",
    )
    generated_image_urls: List[str] = Field(
        default_factory=list,
        description="Public URLs for diagrams generated on this assistant turn.",
    )

class KeyEntities(BaseModel):
    """Extracted entities from a chat session."""
    name: Optional[str] = None
    class_level: Optional[str] = None
    topic: Optional[str] = None
    identified_gaps: Optional[str] = None
    others: Optional[Dict[str, Any]] = None

    @field_validator("name", "class_level", "topic", "identified_gaps", mode="before")
    @classmethod
    def _coerce_text_fields(cls, value: Any) -> Optional[str]:
        """Handle legacy list-shaped values stored in MongoDB."""
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return "; ".join(cleaned) if cleaned else None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    @field_validator("others", mode="before")
    @classmethod
    def _coerce_others(cls, value: Any) -> Optional[Dict[str, Any]]:
        """Normalize legacy non-dict variants for `others`."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            if not value:
                return None
            return {"items": value}
        return {"value": value}

class StructuredSummary(BaseModel):
    """Structured compaction of a conversation turn."""
    summary: str = Field(..., description="A 3-5 sentence overview of the conversation.")
    key_entities: Optional[KeyEntities] = Field(None, description="Key entities like student name, topic, and learning gaps.")
    user_preferences: Optional[str] = Field(None, description="User mentioned preferences or constraints.")

    @field_validator("user_preferences", mode="before")
    @classmethod
    def _coerce_user_preferences(cls, value: Any) -> Optional[str]:
        """Handle legacy list payloads for user preferences."""
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return "; ".join(cleaned) if cleaned else None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

class ChatSession(Document):
    """Document model for a chat session."""
    user_id: str
    session_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    structured_summary: Optional[StructuredSummary] = None
    messages: List[ChatMessage] = []
    uploaded_images: List[UploadedImageRecord] = Field(default_factory=list)
    generated_images: List[GeneratedImageRecord] = Field(default_factory=list)
    is_summarizing: bool = False
    created_at: datetime = Field(default_factory=get_ist_now)
    updated_at: datetime = Field(default_factory=get_ist_now)

    class Settings:
        name = "chat_sessions"
        indexes = [
            "user_id",
            IndexModel([("session_id", ASCENDING)], name="session_id_index", unique=True),
        ]
    
    async def add_message(
        self,
        role: str,
        text: str,
        *,
        has_image_input: bool = False,
        image_input_ids: Optional[List[str]] = None,
        has_generated_image: bool = False,
        generated_image_ids: Optional[List[str]] = None,
        generated_image_urls: Optional[List[str]] = None,
    ):
        """Helper to add a message and update timestamp atomically."""
        message = ChatMessage(
            role=role,
            text=text,
            has_image_input=has_image_input,
            image_input_ids=list(image_input_ids or []),
            has_generated_image=has_generated_image,
            generated_image_ids=list(generated_image_ids or []),
            generated_image_urls=list(generated_image_urls or []),
        )
        # Use atomic update to prevent race conditions
        await self.update(
            {"$push": {"messages": message}, "$set": {"updated_at": get_ist_now()}}
        )
        # Update local instance to reflect changes (optional but good for consistency)
        self.messages.append(message)
        self.updated_at = get_ist_now()
