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
def is_greeting(text: str) -> bool:
    """Check if a string is a common greeting, ignoring repeated characters and common variations."""
    import re
    
    # 1. Clean and normalize
    s = text.lower().strip().rstrip("!?. ")
    if not s:
        return False
        
    # 2. Collapse repeated characters: 'hii' -> 'hi', 'hellooo' -> 'helo'
    collapsed = re.sub(r'(.)\1+', r'\1', s)
    
    # 3. Handle multiple words: 'hi hi' -> ['hi', 'hi']
    words = collapsed.split()
    
    # Collapsed baseline keywords (manual mapping for de-repeated versions)
    # These cover English, Hindi (transliterated and native), and some common others
    greeting_baselines = {
        "hi", "helo", "hey", "thank", "thx", "thanx", "gretings",
        "नमस्ते", "हेलो", "नमस्ते", "शुक्रिया", "धन्यवाद",
        "ok", "okay", "alright", "sure", "fine", "nice", "great", "awesome",
        "yep", "yes", "no", "bye", "godbye", "k", "vadiya"
    }
    
    # Also handle common multi-word greetings/phrases
    multi_word_baselines = ["thank you", "how are you", "whats up", "kaise ho"]
    collapsed_multi = " ".join(words)
    if any(mw in collapsed_multi for mw in multi_word_baselines):
        return True
    
    # If all words in the query are in the greeting baselines, it's likely a greeting
    if all(word in greeting_baselines for word in words) and len(words) > 0:
        return True
            
    return False
