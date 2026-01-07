"""ReAct (Reasoning + Acting) Agent implementation."""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings
from state import ConversationTurn
from tools import ToolRegistry

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I couldn't find a definitive answer in the curriculum materials. Could you please provide more details or rephrase your question?"

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.
    
    Uses an LLM to reason about a query, select tools, execute them,
    and synthesize an answer.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        tool_registry: ToolRegistry,
        max_iterations: Optional[int] = None,
        **kwargs,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations or (settings.max_iterations if settings else 5)
        self.enforce_sequential = kwargs.get("enforce_sequential", False)
        
        # Convert custom tools to LangChain tools
        self.tools = []
        for tool in tool_registry.list_tools():
            if hasattr(tool, "to_langchain"):
                # If tool has conversion method
                self.tools.append(tool.to_langchain())
            elif hasattr(tool, "execute"):
                # Custom Tool class -> Wrap in StructuredTool
                from langchain_core.tools import StructuredTool
                from pydantic import create_model, Field
                
                # Build args_schema from tool.parameters_schema
                fields = {}
                for param_name, param_info in tool.parameters_schema.items():
                    # Map types (simplified)
                    param_type = str
                    if param_info.get("type") == "integer":
                        param_type = int
                    elif param_info.get("type") == "boolean":
                        param_type = bool
                    
                    description = param_info.get("description", "")
                    is_required = param_info.get("required", False)
                    
                    if is_required:
                        fields[param_name] = (param_type, Field(description=description))
                    else:
                        fields[param_name] = (Optional[param_type], Field(None, description=description))
                
                ArgsSchema = create_model(f"{tool.name}Schema", **fields)
                
                # Create async wrapper for execute
                async def _wrapper(**kwargs):
                    return await tool.execute(**kwargs)
                
                langchain_tool = StructuredTool.from_function(
                    func=None,
                    coroutine=_wrapper,
                    name=tool.name,
                    description=tool.description,
                    args_schema=ArgsSchema,
                )
                self.tools.append(langchain_tool)
            else:
                # Assume it's already a LangChain tool
                self.tools.append(tool)
                
        self.llm_with_tools = llm.bind_tools(self.tools)
    
    async def run(
        self,
        query: str,
        messages: List[BaseMessage],
        summary: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
        request_filters: Optional[Dict[str, Any]] = None,
        prefilled_observations: Optional[List[Dict[str, Any]]] = None, # NEW
    ) -> Dict[str, Any]:
        """
        Run the agent loop to answer the query.
        """
        messages = self._build_messages(query, messages, summary, session_metadata)
        scratchpad: List[Dict[str, str]] = []
        
        # Inject prefilled observations (Proactive RAG)
        if prefilled_observations:
            logger.info("Injecting %d prefilled observations", len(prefilled_observations))
            from langchain_core.messages import AIMessage, ToolMessage
            import uuid
            
            tool_calls = []
            obs_messages = []
            for obs in prefilled_observations:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls.append({
                    "name": obs["tool"],
                    "args": obs["args"],
                    "id": call_id
                })
                obs_messages.append(ToolMessage(
                    content=str(obs["observation"]),
                    tool_call_id=call_id,
                    name=obs["tool"]
                ))
                # Record in scratchpad so citations work
                scratchpad.append({
                    "iteration": 0,
                    "thought": "Proactive retrieval...",
                    "action": obs["tool"],
                    "action_input": str(obs["args"]),
                    "observation": obs["observation"]
                })
            
            # Add simulated calls to history
            messages.append(AIMessage(content="", tool_calls=tool_calls))
            messages.extend(obs_messages)
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info("Agent iteration %d/%d", iteration, self.max_iterations)
            
            # 1. Call LLM
            llm_start = asyncio.get_event_loop().time()
            response = await self.llm_with_tools.ainvoke(messages)
            llm_duration = asyncio.get_event_loop().time() - llm_start
            logger.info("Agent iteration %d: LLM call took %.3fs", iteration, llm_duration)
            messages.append(response)
            
            # 2. Check for tool calls
            if not response.tool_calls:
                # No tool calls -> Final answer
                logger.info("Agent finished with final answer")
                return {
                    "answer": response.content,
                    "reasoning_chain": scratchpad,
                    "iterations": iteration,
                }
            
            # 3. Execute tools
            tool_coroutines = []
            tool_metadatas = []
            
            # Detect tool conflict (programmatic rail)
            tool_names_in_call = [tc["name"] for tc in response.tool_calls]
            
            has_retrieval = "retrieve_documents" in tool_names_in_call
            has_web_search = "web_search" in tool_names_in_call

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Clean tool_args: Remove None values or empty strings
                tool_args = {k: v for k, v in tool_args.items() if v is not None and v != ""}
                
                # Check for parallel tool conflict (Only if enforcement is enabled)
                if self.enforce_sequential and tool_name == "web_search" and has_retrieval:
                    logger.warning("Agent attempted parallel RAG and Web Search. Postponing Web Search per strict sequential policy.")
                    
                    # Intercept web_search and return a postponement message
                    async def _postpone_web_search():
                        return "ERROR: Web search cannot be used in parallel with 'retrieve_documents'. Please review the results of the curriculum search below first. If they are insufficient, you may use 'web_search' in the NEXT turn."
                    
                    tool_coroutines.append(_postpone_web_search())
                    tool_metadatas.append({"name": tool_name, "args": tool_args, "id": tool_id})
                    continue

                # Inject filters into retrieval tool
                if tool_name == "retrieve_documents":
                    # ONLY use filters provided in the request body
                    if request_filters:
                        # Strip additionalProp1 (OpenAPI placeholder)
                        clean_filters = {k: v for k, v in request_filters.items() if k != "additionalProp1"}
                        if clean_filters:
                            tool_args["filters"] = clean_filters
                
                logger.info("Tool Call: %s(%s)", tool_name, tool_args)
                
                # Define execution wrapper
                async def _execute_tool(t_name, t_args):
                    try:
                        # Use our custom registry directly to avoid LangChain wrapper issues
                        tool = self.tool_registry.get(t_name)
                        return await tool.execute(**t_args)
                    except Exception as exc:
                        logger.error("Tool execution failed: %s", exc)
                        return f"Error: {str(exc)}"

                tool_coroutines.append(_execute_tool(tool_name, tool_args))
                tool_metadatas.append({"name": tool_name, "args": tool_args, "id": tool_id})

            # Execute all tool calls concurrently
            if tool_coroutines:
                tool_start = asyncio.get_event_loop().time()
                results = await asyncio.gather(*tool_coroutines)
                tool_duration = asyncio.get_event_loop().time() - tool_start
                logger.info("[TRACE] ReActAgent: Gathered %d tool responses in %.3fs.", len(results), tool_duration)
                
                # Process results and update state
                for meta, observation in zip(tool_metadatas, results):
                    # Record in scratchpad
                    scratchpad.append({
                        "iteration": iteration,
                        "thought": "Using tool...", 
                        "action": meta["name"],
                        "action_input": str(meta["args"]),
                        "observation": observation,
                        "duration": tool_duration # Record duration in scratchpad for debugging
                    })
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=str(observation),
                        tool_call_id=meta["id"],
                        name=meta["name"]
                    ))
        
        # Max iterations reached
        logger.warning("Max iterations (%d) reached", self.max_iterations)
        
        # Phase 0: Dynamic Reasoning Synthesis
        # If we found nothing, use fallback. If we found something in scratchpad, try to answer.
        has_observations = any(turn.get("observation") and "NO_DOCS_FOUND" not in str(turn.get("observation")) for turn in scratchpad)
        
        if has_observations:
            logger.info("Attempting to synthesize answer from partial observations after reaching max iterations.")
            synthesis_prompt = (
                "You reached the maximum number of thought steps. Based on the information you collected so far, "
                "provide the most complete answer possible. If you still don't have enough info, be honest but helpful."
            )
            messages.append(HumanMessage(content=synthesis_prompt))
            final_resp = await self.llm.ainvoke(messages)
            return {
                "answer": final_resp.content,
                "reasoning_chain": scratchpad,
                "iterations": self.max_iterations,
            }

        return {
            "answer": FALLBACK_MESSAGE,
            "reasoning_chain": scratchpad,
            "iterations": self.max_iterations,
        }

    def _build_messages(
        self, 
        query: str, 
        history: List[BaseMessage],
        summary: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Build the initial message list incorporating summary and trimmed history."""
        messages = []
        
        # Format metadata for context
        context_str = ""
        if session_metadata:
            context_items = []
            if session_metadata.get("class_name"):
                context_items.append(f"Class: {session_metadata['class_name']}")
            if session_metadata.get("subject"):
                context_items.append(f"Subject: {session_metadata['subject']}")
            if session_metadata.get("topics"):
                context_items.append(f"Topic: {session_metadata['topics']}")
            
            if context_items:
                context_str = "\nCONTEXT:\n" + "\n".join(context_items)
        
        # System prompt incorporating summary (Phase 5)
        summary_text = f"\n\nCONVERSATION SUMMARY:\n{summary}" if summary else ""
        
        if hasattr(self, "build_system_prompt"):
            system_prompt = self.build_system_prompt(query, session_metadata) + summary_text
        else:
            system_prompt = (
                "You are an educational AI assistant. Answer the user's question directly using the retrieved documents.\n"
                "Rules:\n"
                "1. Use 'retrieve_documents' to find information.\n"
                "2. Provide a direct answer. Do NOT explain the retrieval or search process.\n"
                "3. Use the conversation context (e.g., class level, subject) to tailor the complexity and focus of your answer, but do not explicitly mention the context unless necessary.\n"
                "4. If the query is vague, ask for clarification.\n"
                "5. Do NOT mention specific source filenames or IDs in your text response. Citations will be added automatically.\n"
                "6. Do NOT use general knowledge. If the info is not in the documents, say you don't know.\n"
                "7. Use MathJax for ALL mathematical formulas and expressions (e.g., \\( x^2 \\) for inline, \\[ \\frac{a}{b} \\] for block)."
                f"{context_str}"
                f"{summary_text}"
            )
        messages.append(SystemMessage(content=system_prompt))
        
        # History (Phase 0/5: Already trimmed by MemoryService)
        messages.extend(history)
                
        # Current query
        messages.append(HumanMessage(content=query))
        
        return messages


__all__ = ["ReActAgent"]
