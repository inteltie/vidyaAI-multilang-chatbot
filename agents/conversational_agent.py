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
        
        # Detection of Session Restart
        is_restart = state.get("is_session_restart", False)
        prefix = "Welcome back! " if is_restart else ""
        
        # Template-based responses for common cases
        if any(word in query for word in ["thanks", "thank you", "thx"]):
            response = f"{prefix}I'm glad I could help! Feel free to ask if you have more questions."
        elif any(word in query for word in ["bye", "goodbye", "see you"]):
            response = f"{prefix}Goodbye! Happy learning! 📚"
        elif any(phrase in query for phrase in ["solved", "clear now", "got it", "understood"]):
            response = f"{prefix}Great! I'm here whenever you need help with your studies. Is there anything else you'd like to know?"
        else:
            # Use LLM with history and summary for personalized responses
            history = state.get("conversation_history", [])
            summary = state.get("session_metadata", {}).get("summary", "")
            
            history_text = ""
            for m in history[-12:]: 
                role = "STUDENT" if m.type == "human" else "VIDYA"
                history_text += f"{role}: {m.content}\n"
            
            
            # Check if this is truly the first interaction
            has_history = len(history) > 0
            
            target_lang = state.get("language", "en")
            prompt = (
                f"You are Vidya, a friendly and helpful educational assistant. "
                f"Respond naturally to the student's message in **{target_lang}**. "
                f"IMPORTANT: Use the history below to see if the student shared their name and use it.\n"
                f"{'NOTICE: This is a returning student after some time away. Welcome them back warmly.' if is_restart else ''}\n\n"
                f"Conversation Summary: {summary}\n"
                f"Recent Interaction History:\n{history_text}\n"
                f"Latest Message from Student: {state['query']}\n\n"
                f"Response Guidelines:\n"
                f"- Be warm and personalized.\n"
                f"- Your response MUST be in **{target_lang}**.\n"
                f"- If the student shared their name earlier, use it.\n"
                f"- {'CRITICAL: This is MID-CONVERSATION (history exists). DO NOT greet with Hello/Hi/Namaste. Just respond naturally to their message.' if has_history else 'This is the FIRST message. Greet warmly and ask how you can help.'}\n"
                f"- Keep the response brief and encouraging (under 100 tokens)."
            )
            resp = await self._llm.ainvoke(prompt)
            response = resp.content.strip()
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        
        state["response"] = response
        logger.info("Conversational agent handled query%s", " (session restart)" if is_restart else "")
        return state
