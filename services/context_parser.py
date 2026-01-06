"""Service for parsing and extracting session context from conversations."""

from __future__ import annotations

import logging
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from state import ConversationTurn, SessionMetadata

logger = logging.getLogger(__name__)


class ExtractedMetadata(BaseModel):
    """Educational metadata extracted from text."""
    class_level: Optional[str] = Field(None, description="Class level, e.g., '10', '12', 'Class 9'")
    subject: Optional[str] = Field(None, description="Subject, e.g., 'Math', 'Physics'")
    chapter: Optional[str] = Field(None, description="Chapter or topic, e.g., 'Quadratic Equations'")
    last_topic: Optional[str] = Field(None, description="Brief summary of what was being discussed")
    lecture_id: Optional[str] = Field(None, description="Specific lecture ID if mentioned")


class NextStep(BaseModel):
    """Next step in the context gathering process."""
    parsed: ExtractedMetadata = Field(description="Metadata extracted from the user's reply")
    still_missing: List[str] = Field(default_factory=list, description="List of fields still missing")
    next_question: str = Field(description="The next question to ask the user")


class ContextParser:
    """Parse session metadata from conversation history and user replies."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._extractor = llm.with_structured_output(ExtractedMetadata)
        self._next_step_generator = llm.with_structured_output(NextStep)
        
    def _format_history(self, history: List[Union[ConversationTurn, BaseMessage]], limit: int = 6) -> str:
        """Format history into role: content text, handling both dicts and objects."""
        from services.utils import format_history
        return format_history(history, limit)

    async def extract_from_history(self, history: List[Union[ConversationTurn, BaseMessage]]) -> SessionMetadata:
        """Extract session metadata from conversation history."""
        if not history:
            return {}

        history_text = self._format_history(history, limit=6)
        prompt = (
            "You are analyzing a conversation to extract educational context metadata.\n"
            "Scan the conversation and extract the following information if mentioned:\n"
            "- class_level\n"
            "- subject\n"
            "- chapter\n"
            "- lecture_id\n"
            "- last_topic\n\n"
            f"Conversation:\n{history_text}"
        )

        try:
            result: ExtractedMetadata = await self._extractor.ainvoke(prompt)
            return {
                "class_level": result.class_level,
                "subject": result.subject,
                "chapter": result.chapter,
                "lecture_id": result.lecture_id,
                "last_topic": result.last_topic,
            }
        except Exception as exc:
            logger.warning("Failed to extract session metadata from history: %s", exc)
            return {}

    async def extract_from_query(self, query: str, history: List[Union[ConversationTurn, BaseMessage]]) -> SessionMetadata:
        """Extract metadata from the current query."""
        history_text = self._format_history(history, limit=3)
        prompt = (
            "You are analyzing a user query to extract educational context metadata.\n"
            "Extract class_level, subject, chapter, and lecture_id if explicitly mentioned.\n\n"
            f"Previous context:\n{history_text}\n\n"
            f"Current query: {query}"
        )

        try:
            result: ExtractedMetadata = await self._extractor.ainvoke(prompt)
            return {
                "class_level": result.class_level,
                "subject": result.subject,
                "chapter": result.chapter,
                "lecture_id": result.lecture_id,
            }
        except Exception as exc:
            logger.warning("Failed to extract metadata from query: %s", exc)
            return {}

    async def parse_context_reply(self, query: str, history: List[Union[ConversationTurn, BaseMessage]]) -> SessionMetadata:
        """Parse a natural language context reply into structured metadata."""
        history_text = self._format_history(history, limit=3)
        prompt = (
            "The user was asked to provide subject/chapter/class information and has responded.\n"
            "Parse their response and extract metadata.\n"
            "Be flexible with natural language.\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            "- subject: e.g., 'Math', 'Physics', 'Biology', 'Stocks'\n"
            "- chapter OR topic: e.g., 'Quadratic Equations', 'Chapter 5', 'Photosynthesis', 'Forex'\n"
            "- lecture_id: ONLY if explicitly mentioned (e.g., 'lecture 140', 'session 76')\n\n"
            "IMPORTANT FIELD MAPPING:\n"
            "- If user mentions 'class', 'class name', 'batch', or 'grade' → extract as class_level\n"
            "- If user mentions 'topic', 'module' → extract as chapter\n"
            "- Be flexible with natural language like 'subject is stocks and class name is stock market batch'\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            f"User's response: {query}\n\n"
            "Return ONLY a JSON object with these fields (use null for values not provided):\n"
            '{"class_level": "...", "subject": "...", "chapter": "...", "lecture_id": "..."}\n\n'
            "JSON:"
        )

        try:
            result: ExtractedMetadata = await self._extractor.ainvoke(prompt)
            return {
                "class_level": result.class_level,
                "subject": result.subject,
                "chapter": result.chapter,
                "lecture_id": result.lecture_id,
            }
        except Exception as exc:
            logger.warning("Failed to parse context reply: %s", exc)
            return {}

    async def parse_and_ask(
        self,
        query: str,
        history: List[Union[ConversationTurn, BaseMessage]],
        missing_fields: List[str],
        attempts: int,
    ) -> tuple[SessionMetadata, str]:
        """OPTIMIZED: Parse user reply AND generate next question in ONE LLM call."""
        history_text = self._format_history(history, limit=6)
        
        prompt = (
            "You are helping extract educational context from a student's reply.\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            f"Student's latest reply: {query}\n\n"
            f"Still missing: {', '.join(missing_fields)}\n"
            f"Attempt number: {attempts + 1}\n\n"
            "TASK 1: Parse the student's reply and extract any metadata (class, subject, chapter).\n"
            "TASK 2: Generate a follow-up question for any still-missing fields.\n"
        )
        
        if attempts == 0:
            prompt += "- Be friendly and clear\n"
        elif attempts == 1:
            prompt += "- Acknowledge what they said\n- Gently ask for missing info\n"
        else:
            prompt += "- Offer to help anyway\n- Ask for their actual question\n"
            
        try:
            result: NextStep = await self._next_step_generator.ainvoke(prompt)
            
            metadata: SessionMetadata = {
                "class_level": result.parsed.class_level,
                "subject": result.parsed.subject,
                "chapter": result.parsed.chapter,
            }
            
            return metadata, result.next_question
            
        except Exception as exc:
            logger.warning("Failed to parse_and_ask: %s", exc)
            return {}, f"Could you tell me which {' and '.join(missing_fields)} you're studying?"



__all__ = ["ContextParser"]

