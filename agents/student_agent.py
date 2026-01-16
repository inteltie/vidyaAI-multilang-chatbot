"""Student Agent with ReAct reasoning for educational queries."""

import logging
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from agents.react_agent import ReActAgent, FALLBACK_MESSAGE
from tools import ToolRegistry, RetrievalTool
from tools.web_search_tool import WebSearchTool
from services.retriever import RetrieverService
from services.citation_service import CitationService
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
    ):
        self.llm = llm
        self.retriever = retriever
        self.enable_web_search = enable_web_search
        
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
            enforce_sequential=False,
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
5. **Citations**: Use labels like `[Source 1]`, `[Source 2]` at the end of relevant sentences to cite your sources.
6. **Target Language [STRICT]**: {target_lang}.
7. **Efficiency**: {efficiency_instruction}
8. **LOCAL KNOWLEDGE ONLY [STRICT]**: Never mention external websites or links. Use ONLY information from local documents.
9. **BREVITY (MANDATORY)**: Keep your response concise (50-100 tokens). Unless the user asks for more detail, provide only core information.
{correction_instruction}

HOW TO RESPOND:
- Provide your response in {target_lang}, strictly embodying **{identity['name']}** through the rules above.
- Cite sources using `[Source X]` format.
- Stick to the **50-100 token limit** unless detail is specifically requested.
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
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process student query with educational focus.
        
        The agent will:
        - Understand what the student wants to learn
        - Search for relevant educational content
        - Provide clear explanations with examples
        - Encourage learning and understanding
        """
        query = state["query_en"]
        history = state.get("conversation_history", [])
        
        logger.info("Student Agent processing query: %s", query[:100])
        
        # Get target language
        target_lang = state.get("language", "en")
        
        # Run ReAct reasoning loop with student context and target language
        self.react_agent.build_system_prompt = lambda q, s: self._build_student_system_prompt(q, s, target_lang, state)
        
        # Set correction flag for ReAct trace if validation results exist
        val_results = state.get("validation_results")
        if val_results and not val_results.get("is_valid"):
            state["is_correction"] = True
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
                state["response"] = result["answer"]
                
                # Extract citations from reasoning chain
                # source_documents comes from state["documents"] (retrieved in RetrieveDocumentsNode)
                source_docs = state.get("documents", [])
                citations = CitationService.extract_citations(
                    result.get("reasoning_chain", []), 
                    source_docs,
                    min_score=0.4
                )
                if citations:
                    state["citations"] = citations
                
                state["llm_calls"] = state.get("llm_calls", 0) + result.get("iterations", 0)
                
                # Mark final language
                if target_lang != "en" and result["answer"] != FALLBACK_MESSAGE:
                    state["final_language"] = target_lang
            else:
                citations = []
            
            logger.info(
                "Student Agent completed with %d iterations, %d citations",
                result.get("iterations", 0),
                len(citations),
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
            state["response"] = FALLBACK_MESSAGE
            state["llm_calls"] = 0
            state["citations"] = []
        
        return state
    



__all__ = ["StudentAgent"]
