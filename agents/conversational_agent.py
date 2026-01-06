"""Conversational agent for handling acknowledgments and social interactions."""

import logging
from langchain_openai import ChatOpenAI
from state import AgentState

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Handles acknowledgments, greetings, and social interactions."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Generate friendly conversational response."""
        query = state["query"].lower()
        
        # Template-based responses for common cases
        if any(word in query for word in ["thanks", "thank you", "thx"]):
            response = "I'm glad I could help! Feel free to ask if you have more questions."
        elif any(word in query for word in ["bye", "goodbye", "see you"]):
            response = "Goodbye! Happy learning! 📚"
        elif any(phrase in query for phrase in ["solved", "clear now", "got it", "understood"]):
            response = "Great! I'm here whenever you need help with your studies. Is there anything else you'd like to know?"
        elif any(word in query for word in ["hello", "hi", "hey"]):
            response = "Hello! How can I help you with your studies today?"
        else:
            # Use LLM for other conversational responses
            prompt = (
                f"Respond naturally and briefly to this student message: '{state['query']}'\n"
                "Keep it friendly and educational. Offer to help if appropriate."
            )
            resp = await self._llm.ainvoke(prompt)
            response = resp.content.strip()
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        
        state["response"] = response
        logger.info("Conversational agent handled query")
        return state
