"""ReAct (Reasoning + Acting) Agent implementation."""

import json
import logging
import asyncio
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings
from services.costing import (
    merge_usage_by_component,
    parse_web_search_result,
    patch_component_usage,
)
from services.prompt_policy import GLOBAL_GUARDRAIL_RULES
from state import ConversationTurn
from tools import ToolRegistry

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I couldn't find a definitive answer in the curriculum materials. Could you please provide more details or rephrase your question?"


def _add_usage(
    usage_map: Dict[str, Dict[str, int]],
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    if not model_name:
        return
    row = usage_map.setdefault(model_name, {"input": 0, "output": 0})
    row["input"] += int(input_tokens or 0)
    row["output"] += int(output_tokens or 0)


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
        # When True, _build_messages keeps a longer history slice (8 vs 4) even
        # if a session summary is available. Cross-questioning needs the prior
        # Socratic question and the student's latest reply to remain in context.
        self._is_interactive = bool(kwargs.get("is_interactive", False))
        
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
        system_prompt_builder: Optional[Callable[[str, Optional[Dict[str, Any]]], str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent loop to answer the query.

        system_prompt_builder: Per-invocation system prompt override. Use this
        instead of mutating ``self.build_system_prompt`` when the same
        ReActAgent instance may serve concurrent requests (builds a prompt from
        ``query`` and ``session_metadata``).
        """
        # Calculate history tokens before modifying messages
        try:
            history_tokens = self.llm.get_num_tokens_from_messages(messages)
            logger.info("[TOKEN_USAGE] Context: chat_history_tokens=%d", history_tokens)
        except Exception as e:
            logger.debug("Failed to calculate history tokens: %s", e)

        messages = self._build_messages(
            query, messages, summary, session_metadata, system_prompt_builder
        )
        scratchpad: List[Dict[str, str]] = []
        
        total_input_tokens = 0
        total_output_tokens = 0
        token_usage_by_model: Dict[str, Dict[str, int]] = {}
        usage_by_component: Dict[str, Dict[str, Any]] = {}
        web_search_calls = 0
        main_model = getattr(self.llm, "model_name", None) or "unknown"
        web_search_model = settings.web_search_model_name or "gpt-4o-mini"
        
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
            
            # Calculate document tokens
            try:
                all_obs_text = "\n".join([str(obs["observation"]) for obs in prefilled_observations])
                doc_tokens = self.llm.get_num_tokens(all_obs_text)
                logger.info("[TOKEN_USAGE] Context: retrieved_documents_tokens=%d (Count: %d)", doc_tokens, len(prefilled_observations))
            except Exception as e:
                logger.debug("Failed to calculate document tokens: %s", e)
            
            # Add simulated calls to history
            messages.append(AIMessage(content="", tool_calls=tool_calls))
            messages.extend(obs_messages)

        prefilled_tool_names = {str((row or {}).get("tool") or "") for row in (prefilled_observations or [])}
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info("Agent iteration %d/%d", iteration, self.max_iterations)
            
            # 1. Call LLM
            llm_start = asyncio.get_event_loop().time()
            response = await self.llm_with_tools.ainvoke(messages, config={"max_tokens": settings.main_response_tokens})
            
            # Log token usage
            usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage", {})
            if usage:
                 i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                 o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                 total_input_tokens += i_tokens
                 total_output_tokens += o_tokens
                 _add_usage(token_usage_by_model, main_model, i_tokens, o_tokens)
                 usage_by_component = merge_usage_by_component(
                     usage_by_component,
                     patch_component_usage("chat", main_model, i_tokens, o_tokens).get(
                         "usage_by_component", {}
                     ),
                 )
                 logger.info(
                     "[TOKEN_USAGE] ReActAgent (Iteration %d): input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                     iteration,
                     i_tokens,
                     o_tokens,
                     usage.get("total_tokens"),
                     self.llm.model_name
                 )
            
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
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "token_usage_by_model": token_usage_by_model,
                    "usage_by_component": usage_by_component,
                    "web_search_calls": web_search_calls,
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

                # Avoid redundant retrieval/web calls when prefilled context already exists.
                if prefilled_tool_names and tool_name in {"retrieve_documents", "web_search"}:
                    logger.info(
                        "Skipping redundant %s call because prefilled observations exist.",
                        tool_name,
                    )

                    async def _skip_redundant_tool(name: str = tool_name) -> str:
                        return (
                            f"SKIPPED_REDUNDANT_{name.upper()}: "
                            "Use existing context and answer directly unless user explicitly asks for fresh lookup."
                        )

                    tool_coroutines.append(_skip_redundant_tool())
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
                    if meta["name"] == "web_search":
                        observation_str, i_t, o_t, billable = parse_web_search_result(
                            observation
                        )
                        total_input_tokens += i_t
                        total_output_tokens += o_t
                        _add_usage(token_usage_by_model, web_search_model, i_t, o_t)
                        usage_by_component = merge_usage_by_component(
                            usage_by_component,
                            patch_component_usage(
                                "web_search",
                                web_search_model,
                                i_t,
                                o_t,
                                billable_web_call=billable,
                            ).get("usage_by_component", {}),
                        )
                        if billable:
                            web_search_calls += 1
                    elif isinstance(observation, tuple) and len(observation) >= 3:
                        observation_str = str(observation[0])
                    else:
                        observation_str = str(observation)

                    # Record in scratchpad
                    scratchpad.append({
                        "iteration": iteration,
                        "thought": "Using tool...", 
                        "action": meta["name"],
                        "action_input": str(meta["args"]),
                        "observation": observation_str,
                        "duration": tool_duration # Record duration in scratchpad for debugging
                    })
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=observation_str,
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
                f"{GLOBAL_GUARDRAIL_RULES}"
            )
            messages.append(HumanMessage(content=synthesis_prompt))
            final_resp = await self.llm.ainvoke(messages, config={"max_tokens": settings.main_response_tokens})
            
            # Log token usage
            usage = getattr(final_resp, "usage_metadata", None) or getattr(final_resp, "response_metadata", {}).get("token_usage", {})
            if usage:
                 i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                 o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                 total_input_tokens += i_tokens
                 total_output_tokens += o_tokens
                 _add_usage(token_usage_by_model, main_model, i_tokens, o_tokens)
                 usage_by_component = merge_usage_by_component(
                     usage_by_component,
                     patch_component_usage("chat", main_model, i_tokens, o_tokens).get(
                         "usage_by_component", {}
                     ),
                 )
                 logger.info(
                     "[TOKEN_USAGE] ReActAgent (Synthesis): input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                     i_tokens,
                     o_tokens,
                     usage.get("total_tokens"),
                     self.llm.model_name
                 )
            
            return {
                "answer": final_resp.content,
                "reasoning_chain": scratchpad,
                "iterations": self.max_iterations,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "token_usage_by_model": token_usage_by_model,
                "usage_by_component": usage_by_component,
                "web_search_calls": web_search_calls,
            }

        return {
            "answer": FALLBACK_MESSAGE,
            "reasoning_chain": scratchpad,
            "iterations": self.max_iterations,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "token_usage_by_model": token_usage_by_model,
            "usage_by_component": usage_by_component,
            "web_search_calls": web_search_calls,
        }

    def _build_messages(
        self, 
        query: str, 
        history: List[BaseMessage],
        summary: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
        system_prompt_builder: Optional[Callable[[str, Optional[Dict[str, Any]]], str]] = None,
    ) -> List[Any]:
        """Build the initial message list incorporating summary and trimmed history."""
        messages = []
        structured_summary = (session_metadata or {}).get("structured_summary")
        
        # Accept both typed Pydantic models and legacy dict payloads.
        structured_summary_data: Dict[str, Any] = {}
        if structured_summary:
            if isinstance(structured_summary, Mapping):
                structured_summary_data = dict(structured_summary)
            elif hasattr(structured_summary, "model_dump"):
                structured_summary_data = structured_summary.model_dump()
            else:
                structured_summary_data = {
                    "summary": getattr(structured_summary, "summary", ""),
                    "key_entities": getattr(structured_summary, "key_entities", {}),
                    "user_preferences": getattr(structured_summary, "user_preferences", ""),
                }
        
        # Format metadata for context
        context_items = []
        if session_metadata:
            if session_metadata.get("class_name"):
                context_items.append(f"Class: {session_metadata['class_name']}")
            if session_metadata.get("subject"):
                context_items.append(f"Subject: {session_metadata['subject']}")
            if session_metadata.get("topics"):
                context_items.append(f"Topic: {session_metadata['topics']}")
        
        # 1. System Prompt
        # Use structured summary if available, fallback to simple summary
        summary_section = ""
        if structured_summary_data:
            key_entities = structured_summary_data.get("key_entities") or {}
            if hasattr(key_entities, "model_dump"):
                key_entities = key_entities.model_dump()
            elif not isinstance(key_entities, Mapping):
                key_entities = {}

            try:
                key_entities_text = json.dumps(key_entities)
            except TypeError:
                try:
                    key_entities_text = json.dumps(key_entities, default=str)
                except Exception:
                    key_entities_text = str(key_entities)
            
            user_preferences = structured_summary_data.get("user_preferences")
            if isinstance(user_preferences, list):
                user_preferences_text = ", ".join(str(item) for item in user_preferences if str(item).strip())
            elif user_preferences is None:
                user_preferences_text = ""
            else:
                user_preferences_text = str(user_preferences)
            user_preferences_text = user_preferences_text.strip()

            summary_section = (
                "\n\n### CONVERSATION CONTEXT (RECONSTRUCTED MEMORY):\n"
                f"- **Overview**: {structured_summary_data.get('summary', '')}\n"
                f"- **Key Entities**: {key_entities_text}\n"
                f"- **User Preferences**: {user_preferences_text}\n"
            )
        elif summary:
            summary_section = f"\n\nCONVERSATION SUMMARY:\n{summary}"

        context_str = ""
        if context_items:
            context_str = "\nCONTEXT:\n" + "\n".join(context_items)
        
        builder = system_prompt_builder
        if builder is None and hasattr(self, "build_system_prompt"):
            builder = self.build_system_prompt

        if builder is not None:
            system_prompt = builder(query, session_metadata) + summary_section
        else:
            system_prompt = (
                "You are an educational AI. Answer using retrieved docs.\n"
                "Rules:\n"
                "1. Use 'retrieve_documents' for info.\n"
                "2. Answer directly. No retrieval process explanation.\n"
                "3. Tailor complexity to context, don't mention context.\n"
                "4. Ask for clarification if vague.\n"
                "5. Do NOT include any source labels or citations (e.g., [Source 1]) in your text response. Citations are handled automatically.\n"
                "6. No general knowledge. If not in docs, say 'I don't know'.\n"
                "7. Use MathJax for formulas (\\(x\\) inline, \\[y\\] block)."
                f"{GLOBAL_GUARDRAIL_RULES}"
                f"{context_str}"
                f"{summary_section}"
            )
        messages.append(SystemMessage(content=system_prompt))
        
        # 2. History (Aggressive Local Truncation)
        # If we have a summary, we only need the last few messages for immediate context/flow.
        # Otherwise, use the full provided history (which is already limited by MemoryService).
        # Interactive (Socratic) mode keeps a wider window so the prior question
        # and the student's latest answer are always in context for cross-questioning.
        keep = 8 if getattr(self, "_is_interactive", False) else 4
        if (summary or structured_summary) and len(history) > keep:
            logger.info(
                "Applying local truncation: keeping last %d messages due to summary availability (interactive=%s).",
                keep,
                getattr(self, "_is_interactive", False),
            )
            messages.extend(history[-keep:])
        else:
            messages.extend(history)
                
        # 3. Current query
        messages.append(HumanMessage(content=query))
        
        return messages


__all__ = ["ReActAgent"]
