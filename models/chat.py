from typing import List, Optional
from datetime import datetime, timezone, timedelta
from beanie import Document
from pydantic import BaseModel, Field
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

class ChatSession(Document):
    """Document model for a chat session."""
    user_id: str
    session_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
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
