from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from beanie import Document
from pydantic import BaseModel, Field, field_validator
from pymongo import IndexModel, ASCENDING

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current time in IST."""
    return datetime.now(IST)

class ChatMessage(BaseModel):
    """Embedded model for a single chat message."""
    role: str
    text: str
    timestamp: datetime = Field(default_factory=get_ist_now)

class KeyEntities(BaseModel):
    """Extracted entities from a chat session."""
    name: Optional[str] = None
    topic: Optional[str] = None
    identified_gaps: Optional[str] = None
    others: Optional[Dict[str, Any]] = None

    @field_validator("name", "topic", "identified_gaps", mode="before")
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
    is_summarizing: bool = False
    created_at: datetime = Field(default_factory=get_ist_now)
    updated_at: datetime = Field(default_factory=get_ist_now)

    class Settings:
        name = "chat_sessions"
        indexes = [
            "user_id",
            IndexModel([("session_id", ASCENDING)], name="session_id_index", unique=True),
        ]
    
    async def add_message(self, role: str, text: str):
        """Helper to add a message and update timestamp atomically."""
        message = ChatMessage(role=role, text=text)
        # Use atomic update to prevent race conditions
        await self.update(
            {"$push": {"messages": message}, "$set": {"updated_at": get_ist_now()}}
        )
        # Update local instance to reflect changes (optional but good for consistency)
        self.messages.append(message)
        self.updated_at = get_ist_now()
