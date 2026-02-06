"""General knowledge agent for answering broad educational questions."""

import logging
from langchain_openai import ChatOpenAI
from state import AgentState
from config import settings

logger = logging.getLogger(__name__)


class GeneralAgent:
    """Answers general educational questions without RAG retrieval."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
    
    async def __call__(self, state: AgentState) -> dict:
        """Generate educational response from general knowledge."""
        query = state["query_en"]
        user_type = state["user_type"]
        history = state.get("conversation_history", [])
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history[-settings.memory_buffer_size:])
        summary = state.get("session_metadata", {}).get("summary", "")
        
        # Log history tokens
        try:
            # GeneralAgent uses dict-based history in its internal loop, but we can convert to BaseMessages for counting
            from langchain_core.messages import HumanMessage, AIMessage
            messages_for_counting = []
            for t in history[-settings.memory_buffer_size:]:
                if t['role'] == 'user':
                    messages_for_counting.append(HumanMessage(content=t['content']))
                else:
                    messages_for_counting.append(AIMessage(content=t['content']))
            
            history_tokens = self._llm.get_num_tokens_from_messages(messages_for_counting)
            logger.info("[TOKEN_USAGE] Context: chat_history_tokens=%d", history_tokens)
        except Exception as e:
            logger.debug("Failed to calculate history tokens: %s", e)
        
        role_instructions = (
            "Explain clearly and simply, step-by-step, suitable for a student. Use analogies when helpful."
            if user_type == "student"
            else "Provide comprehensive explanation suitable for a teacher, including key points and common pitfalls."
        )
        
        prompt = f"""You are an educational AI.
{role_instructions}

Summary: {summary}
History:
{history_text}

Query: {query}
Respond concisely.
"""
        resp = await self._llm.ainvoke(prompt, config={"max_tokens": settings.main_response_tokens})
        response = resp.content.strip()
        
        updates = {
            "response": response,
            "llm_calls": 1,
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        # Log token usage
        usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
        if usage:
            i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            updates["input_tokens"] = i_tokens
            updates["output_tokens"] = o_tokens
            logger.info(
                "[TOKEN_USAGE] GeneralAgent: input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                i_tokens,
                o_tokens,
                usage.get("total_tokens"),
                self._llm.model_name
            )
            
        logger.info("General agent handled query")
        return updates
