"""Conversational agent for handling acknowledgments and social interactions."""

import logging
from langchain_openai import ChatOpenAI
from state import AgentState
from config import settings

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
            for m in history[-settings.memory_buffer_size:]: 
                role = "STUDENT" if m.type == "human" else "VIDYA"
                history_text += f"{role}: {m.content}\n"
            
            # Log history tokens
            try:
                history_tokens = self._llm.get_num_tokens_from_messages(history[-settings.memory_buffer_size:])
                logger.info("[TOKEN_USAGE] Context: chat_history_tokens=%d", history_tokens)
            except Exception as e:
                logger.debug("Failed to calculate history tokens: %s", e)

            has_history = len(history) > 0
            
            target_lang = state.get("language", "en")
            
            # Different prompts for fresh vs mid-conversation greetings
            if not has_history:
                # Fresh conversation - simple welcome
                prompt = (
                    f"You are Vidya, a friendly educational assistant. "\
                    f"The user just greeted you for the first time. "\
                    f"Respond in **{target_lang}** with a warm, brief welcome.\n\n"\
                    f"User Greeting: {state['query']}\n\n"\
                    f"Rules:\n"\
                    f"- Be warm and welcoming. Use emojis (👋, 📚).\n"\
                    f"- Introduce yourself briefly as Vidya, their learning companion.\n"\
                    f"- Ask how you can help them today.\n"\
                    f"- Keep it brief and inviting (<60 tokens).\n"\
                    f"- DO NOT mention any previous topics or discussions.")
            else:
                # Mid-conversation greeting - reconnect pattern
                prompt = (
                    f"You are Vidya, a friendly educational assistant. "\
                    f"The user is greeting you mid-conversation. "\
                    f"Respond in **{target_lang}** using the 'Acknowledge & Reconnect' pattern:\n\n"\
                    f"1. Greet the user warmly.\n"\
                    f"2. Briefly mention the ongoing topic from the context (e.g., 'We were just discussing {summary if summary else 'your studies'}').\n"\
                    f"3. Ask if they want to continue with that topic or start something new.\n\n"\
                    f"Context Summary: {summary}\n"\
                    f"Recent History Snippet:\n{history_text}\n"\
                    f"User Greeting: {state['query']}\n\n"\
                    f"Rules:\n"\
                    f"- Be warm and human-like. Use emojis (👋, 📚).\n"\
                    f"- NEVER say generic 'I see you are greeting me'.\n"\
                    f"- Keep it brief and inviting (<60 tokens).")
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
