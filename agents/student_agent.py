"""Student Agent with ReAct reasoning for educational queries."""

import logging
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from agents.react_agent import ReActAgent, FALLBACK_MESSAGE
from agents.image_analyzer_agent import ImageAnalyzerAgent
from agents.image_context_gate_agent import ImageContextGateAgent
from services.tutor_context_composer import TutorContextComposer
from services.image_turn_pipeline import prepare_image_turn
from tools import ToolRegistry, RetrievalTool
from tools.web_search_tool import WebSearchTool
from services.retriever import RetrieverService
from services.citation_service import CitationService
from services.prompt_policy import GLOBAL_GUARDRAIL_RULES
from state import AgentState, ConversationTurn
from config import settings

logger = logging.getLogger(__name__)


class StudentAgent:
    """
    Student-focused educational agent using ReAct reasoning.
    
    Helps students:
    - Understand concepts clearly
    - Learn with examples and explanations
    - Solve doubts step-by-step
    - Build knowledge progressively
    
    Uses retrieval-augmented generation with educational tone.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: RetrieverService,
        max_iterations: Optional[int] = None,
        enable_web_search: bool = True,
        image_analyzer: Optional[ImageAnalyzerAgent] = None,
        image_gate: Optional[ImageContextGateAgent] = None,
        tutor_composer: Optional[TutorContextComposer] = None,
        llm_fast: Optional[ChatOpenAI] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.enable_web_search = enable_web_search
        self._image_analyzer = image_analyzer or ImageAnalyzerAgent(llm)
        self._image_gate = image_gate or ImageContextGateAgent(llm_fast)
        self._tutor_composer = tutor_composer or TutorContextComposer()
        
        # Resolve config safely
        real_max_iterations = max_iterations or (settings.max_iterations if settings else 5)
        
        # Create dedicated tool registry for student learning
        self.tool_registry = ToolRegistry()
        self._register_student_tools()
        
        # Create ReAct agent for student learning
        self.react_agent = ReActAgent(
            llm=llm,
            tool_registry=self.tool_registry,
            max_iterations=real_max_iterations,
            enforce_sequential=True,
        )
    
    def _build_student_system_prompt(self, query: str, subjects: List[str], target_lang: str, state: Optional[Dict[str, Any]] = None) -> str:
        subjects_str = ", ".join(subjects) if subjects else "General"
        
        # GRADE-BASED OPERATIONAL IDENTITY
        grade = (state or {}).get("student_grade", "B")
        logger.info("--- [DEBUG] Building StudentAgent prompt for Grade: %s ---", grade)
        
        identities = {
            "A": {
                "name": "The Analytic Architect [PERSONA_ARC_A]",
                "focus": "High-level synthesis, technical depth, and critical inquiry.",
                "rules": [
                    "NEVER start with a dictionary definition. Assume mastery of basics.",
                    "MANDATORY: Use the exact technical term 'Kinetic Impedance' once.",
                    "MANDATORY: End with one technical 'What if...' question.",
                    "Tone: Precise, professional, and intellectually rigorous."
                ]
            },
            "B": {
                "name": "The Structured Scholar [PERSONA_SCH_B]",
                "focus": "Balanced understanding, logical flow, and practical application.",
                "rules": [
                    "Start with a clear definition.",
                    "Use standard academic structure.",
                    "Tone: Clear, helpful, and academically supportive."
                ]
            },
            "C": {
                "name": "The Helpful Neighbor [PERSONA_NEI_C]",
                "focus": "Core comprehension, simplicity, and confidence building.",
                "rules": [
                    "Explain using analogies like 'sandpaper'.",
                    "MANDATORY: Include one clear, concrete real-world example.",
                    "MANDATORY: Include one encouraging sentence: 'This is a great topic to explore!'",
                    "Tone: Patient, warm, and encouraging."
                ]
            },
            "D": {
                "name": "The Foundational Coach [PERSONA_COA_D]",
                "focus": "Extreme simplicity and reassurance.",
                "rules": [
                    "MANDATORY: Include a very simple story or example to illustrate the point.",
                    "MANDATORY: Start and end with 'You've got this!'",
                    "Strictly NO technical jargon or abstract theory.",
                    "Tone: Highly enthusiastic and super simple."
                ]
            }
        }
        
        identity = identities.get(grade, identities["B"])
        identity_rules = "\n".join([f"- {r}" for r in identity["rules"]])

        # PROACTIVE EFFICIENCY RULE
        rag_quality = (state or {}).get("rag_quality", "low")
        efficiency_instruction = ""
        if rag_quality == "high":
            efficiency_instruction = "Highly relevant curriculum documents are already provided. Synthesize your answer IMMEDIATELY. Do NOT call retrieval again."
        
        # CORRECTION FEEDBACK
        correction_instruction = ""
        val_results = (state or {}).get("validation_results")
        if val_results and not val_results.get("is_valid"):
            feedback = val_results.get("feedback")
            correction_instruction = f"\n\n> [!IMPORTANT]\n> **CORRECTION NEEDED**: {feedback}"

        prompt = f"""You are 'Vidya', acting as **{identity['name']}** for a student with Grade {grade}.
Focus: {identity['focus']}

### YOUR OPERATIONAL IDENTITY RULES:
{identity_rules}

1. **EXPLICIT INTENT PRIORITY (CRITICAL)**: Prioritize the student's *current* input over any previous conversation history or summary. Use memory only as a supportive aid to understand the context (e.g., student name, grade level) or to deepen the discussion IF requested.
2. **NO UNPROMPTED RECAPS**: Do not mention or repeat previous topics, questions, or summaries unless the student explicitly asks to "continue", "tell me more", or "expand further".
3. **AMBIGUITY HANDLING**: If the student's message is vague or ambiguous, politely ask for clarification instead of guessing based on history.
4. **NO META-TALK**: Never say "I searched" or "Based on documents".
5. **Citations [STRICT]**: Do NOT include any source labels or citations (e.g., [Source 1]) in your text.
6. **Response Language [STRICT]**: ALWAYS respond in English. (Translation will be handled by a separate process).
7. **Efficiency**: {efficiency_instruction}
8. **NO EXTERNAL LINKS [MANDATORY]**: Never mention or link to external websites, platforms, or tools (e.g., Coursera, Khan Academy, Wikipedia, YouTube). Use ONLY the provided local information. 
9. **BREVITY (MANDATORY)**: Keep your response concise (50-100 tokens). Unless the user asks for more detail, provide only core information.
{correction_instruction}

HOW TO RESPOND:
- Provide your response in English, strictly embodying **{identity['name']}** through the rules above.
- Do NOT use any source labels or citations in the text.
- Stick to the **50-100 token limit** unless detail is specifically requested.
{GLOBAL_GUARDRAIL_RULES}
"""
        return prompt
    
    def _register_student_tools(self):
        """Register tools for student learning."""
        # Retrieval tool for searching learning materials
        retrieval_tool = RetrievalTool(self.retriever)
        self.tool_registry.register(retrieval_tool)
        
        if self.enable_web_search:
            web_search_tool = WebSearchTool()
            self.tool_registry.register(web_search_tool)
            logger.info("--- [VERIFICATION] Student Agent: Web Search is ENABLED ---")
        else:
            logger.info("--- [VERIFICATION] Student Agent: Web Search is STRICTLY DISABLED (RAG-ONLY) ---")
        
        logger.info("Registered %d tools for Student agent", len(self.tool_registry.list_tools()))
    
    async def __call__(self, state: AgentState) -> dict:
        """
        Process student query with educational focus.
        """
        prep: Dict[str, Any] = {}
        has_image_work = bool(
            state.get("images")
            or state.get("remembered_images")
            or state.get("has_images")
        )
        if has_image_work:
            prep = await prepare_image_turn(
                state,
                image_analyzer=self._image_analyzer,
                image_gate=self._image_gate,
                tutor_composer=self._tutor_composer,
                turn_type="opening",
            )

        tutor_input = prep.get("tutor_input")
        query = (
            tutor_input.query_for_react
            if tutor_input
            else state.get("effective_query")
            or state.get("query_en")
            or state.get("query", "")
        )
        history = state.get("conversation_history", [])
        system_extra = tutor_input.system_extra if tutor_input else ""
        ctx_block = str(state.get("context_resolution_block") or "").strip()
        if ctx_block:
            system_extra = f"{system_extra}\n\n{ctx_block}".strip() if system_extra else ctx_block

        logger.info("Student Agent processing query: %s", query[:100])

        # Get target language
        target_lang = state.get("language", "en")

        def system_prompt_builder(q: str, session_meta: Optional[Dict[str, Any]]) -> str:
            subjects = state.get("subjects")
            if not isinstance(subjects, list):
                maybe_subjects = (session_meta or {}).get("subjects", [])
                subjects = maybe_subjects if isinstance(maybe_subjects, list) else []
            base = self._build_student_system_prompt(q, subjects, target_lang, state)
            preferred_fmt = str(state.get("preferred_response_format") or "").strip().lower()
            if preferred_fmt in {"bullets", "paragraph", "steps"}:
                base += (
                    f"\n\nResponse format preference (remembered): {preferred_fmt}. "
                    "Follow it unless the user explicitly asks for a different format on this turn."
                )
            if state.get("wants_visual_output") and (state.get("agent_mode") or "") != "interactive":
                base += (
                    "\n\nThe student asked for a diagram/flowchart/visual explanation. "
                    "Provide a clear structured text explanation with labeled steps or an ASCII/Mermaid-style "
                    "outline they can follow. Mention that interactive mode can generate a full illustration."
                )
            if system_extra:
                if tutor_input:
                    return (
                        f"{base}\n\n"
                        "The student has shared image(s). Use ONLY the visual context below "
                        "for the current question — do not assume content from older uploads.\n"
                        f"{system_extra}"
                    )
                return f"{base}\n\n{system_extra}"
            return base

        self.react_agent.build_system_prompt = system_prompt_builder

        updates = {
            "response": "",
            "citations": [],
            "llm_calls": int(prep.get("llm_calls", 0) or 0),
            "input_tokens": int(prep.get("input_tokens", 0) or 0),
            "output_tokens": int(prep.get("output_tokens", 0) or 0),
            "is_correction": False,
            "agent_findings": prep.get("agent_findings") or state.get("agent_findings") or [],
            "image_inject_level": prep.get("image_inject_level", "none"),
            "image_inject_block": prep.get("image_inject_block", ""),
            "image_gate_reason": prep.get("image_gate_reason", ""),
            "new_uploaded_image_records": prep.get("new_uploaded_image_records") or [],
            "effective_query": prep.get("effective_query") or state.get("effective_query") or "",
        }
        if prep.get("token_usage_by_model"):
            updates["token_usage_by_model"] = prep["token_usage_by_model"]
        if prep.get("usage_by_component"):
            updates["usage_by_component"] = prep["usage_by_component"]
        if prep.get("aux_input_tokens"):
            updates["aux_input_tokens"] = prep["aux_input_tokens"]
            updates["aux_output_tokens"] = prep.get("aux_output_tokens", 0)

        # Set correction flag if validation results exist
        val_results = state.get("validation_results")
        if val_results and not val_results.get("is_valid"):
            updates["is_correction"] = True
            logger.info("Retrying Student Agent with corrective feedback...")
        
        try:
            session_metadata = state.get("session_metadata", {})
            request_filters = state.get("request_filters", {})
            summary = session_metadata.get("summary")
            result = await self.react_agent.run(
                query, 
                history, 
                summary, 
                session_metadata, 
                request_filters,
                prefilled_observations=state.get("prefilled_observations")
            )
            
            # If we have a valid result from the agent
            if result and "answer" in result:
                updates["response"] = result["answer"]
                
                # Extract citations from reasoning chain
                source_docs = state.get("documents", [])
                citations = CitationService.extract_citations(
                    result.get("reasoning_chain", []), 
                    source_docs,
                    min_score=0.4
                )
                if citations:
                    updates["citations"] = citations
                
                prep_llm = int(prep.get("llm_calls", 0) or 0)
                prep_in = int(prep.get("input_tokens", 0) or 0)
                prep_out = int(prep.get("output_tokens", 0) or 0)
                updates["llm_calls"] = prep_llm + int(result.get("iterations", 0) or 0)
                updates["input_tokens"] = prep_in + int(result.get("input_tokens", 0) or 0)
                updates["output_tokens"] = prep_out + int(result.get("output_tokens", 0) or 0)
                updates["token_usage_by_model"] = result.get("token_usage_by_model", {})
                if result.get("usage_by_component"):
                    updates["usage_by_component"] = result.get("usage_by_component")
                if result.get("web_search_calls"):
                    updates["web_search_calls"] = result.get("web_search_calls")
                
                # Separate primary LLM (Mini) tokens from total/aux (Nano) for breakdown tracking
                updates["primary_input_tokens"] = result.get("input_tokens", 0)
                updates["primary_output_tokens"] = result.get("output_tokens", 0)
            
            logger.info(
                "Student Agent completed with %d iterations, %d citations",
                result.get("iterations", 0),
                len(updates["citations"]),
            )
        except Exception as exc:
            logger.error(
                "Student Agent execution failed for user %s, session %s, query: %s - Error: %s",
                state.get("user_id"),
                state.get("user_session_id"),
                query[:100],
                exc,
                exc_info=True
            )
            # Return fallback response
            updates["response"] = FALLBACK_MESSAGE
            updates["llm_calls"] = 0
            updates["citations"] = []
        
        return updates
    



__all__ = ["StudentAgent"]
