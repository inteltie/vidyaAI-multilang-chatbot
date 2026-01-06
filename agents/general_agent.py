"""General knowledge agent for answering broad educational questions."""

import logging
from langchain_openai import ChatOpenAI
from state import AgentState

logger = logging.getLogger(__name__)


class GeneralAgent:
    """Answers general educational questions without RAG retrieval."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Generate educational response from general knowledge."""
        query = state["query_en"]
        user_type = state["user_type"]
        history = state.get("conversation_history", [])
        
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:])
        
        role_instructions = (
            "Explain clearly and simply, step-by-step, suitable for a student. Use analogies when helpful."
            if user_type == "student"
            else "Provide comprehensive explanation suitable for a teacher, including key points and common pitfalls."
        )
        
        prompt = f"""You are an educational assistant.

{role_instructions}

Conversation history:
{history_text}

Student question: {query}

Provide a helpful, educational answer. Keep it concise but informative.
Optionally end with one short follow-up question to encourage learning.
"""
        
        resp = await self._llm.ainvoke(prompt)
        state["response"] = resp.content.strip()
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        
        logger.info("General agent handled query")
        return state
