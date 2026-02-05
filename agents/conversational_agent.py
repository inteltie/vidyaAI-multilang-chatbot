"""Conversational agent for handling acknowledgments and social interactions."""

import logging
from langchain_openai import ChatOpenAI
from state import AgentState

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Handles acknowledgments, greetings, and social interactions."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
    
    async def __call__(self, state: AgentState) -> dict:
        """Generate friendly conversational response."""
        query = state["query"].lower()
        
        # Detection of Session Restart
        is_restart = state.get("is_session_restart", False)
        prefix = "Welcome back! " if is_restart else ""
        
        updates = {
            "response": "",
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        # Template-based responses for common cases
        if any(word in query for word in ["thanks", "thank you", "thx"]):
            updates["response"] = f"{prefix}I'm glad I could help! Feel free to ask if you have more questions."
        elif any(word in query for word in ["bye", "goodbye", "see you"]):
            updates["response"] = f"{prefix}Goodbye! Happy learning! 📚"
        elif any(phrase in query for phrase in ["solved", "clear now", "got it", "understood"]):
            updates["response"] = f"{prefix}Great! I'm here whenever you need help with your studies. Is there anything else you'd like to know?"
        else:
            # Use LLM with history and summary for personalized responses
            history = state.get("conversation_history", [])
            summary = state.get("session_metadata", {}).get("summary", "")
            
            history_text = ""
            for m in history[-12:]: 
                role = "STUDENT" if m.type == "human" else "VIDYA"
                history_text += f"{role}: {m.content}\n"
            
            # Log history tokens
            try:
                history_tokens = self._llm.get_num_tokens_from_messages(history[-12:])
                logger.info("[TOKEN_USAGE] Context: chat_history_tokens=%d", history_tokens)
            except Exception as e:
                logger.debug("Failed to calculate history tokens: %s", e)

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
                f"- Keep the response brief and encouraging (under 100 tokens).")
            from config import settings
            resp = await self._llm.ainvoke(prompt, config={"max_tokens": settings.main_response_tokens})
            updates["response"] = resp.content.strip()
            
            # Log token usage
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
            if usage:
                i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                updates["input_tokens"] = i_tokens
                updates["output_tokens"] = o_tokens
                logger.info(
                    "[TOKEN_USAGE] ConversationalAgent: input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                    i_tokens,
                    o_tokens,
                    usage.get("total_tokens"),
                    self._llm.model_name
                )
            
            updates["llm_calls"] = 1
        
        logger.info("Conversational agent handled query%s", " (session restart)" if is_restart else "")
        return updates
