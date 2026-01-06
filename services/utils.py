"""Shared utility functions for services."""

from typing import List, Union

from langchain_core.messages import BaseMessage, HumanMessage

from state import ConversationTurn


def format_history(
    history: List[Union[ConversationTurn, BaseMessage]], 
    limit: int = 4
) -> str:
    """Format conversation history into role: content text.
    
    Handles both dict-style ConversationTurn and LangChain BaseMessage objects.
    
    Args:
        history: List of conversation turns (dicts or BaseMessage objects)
        limit: Maximum number of recent turns to include
        
    Returns:
        Formatted string with "role: content" on each line
    """
    formatted = []
    for t in history[-limit:]:
        if isinstance(t, dict):
            role = t.get("role", "user")
            content = t.get("content", "")
        else:
            # BaseMessage (HumanMessage, AIMessage, etc.)
            role = "user" if isinstance(t, HumanMessage) else "assistant"
            content = t.content
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)
