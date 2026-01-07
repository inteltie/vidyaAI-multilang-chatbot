"""Interactive Student Agent with step-by-step reasoning for educational queries."""

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


# Socratic tutoring focuses on guided discovery rather than lecturing.


class InteractiveStudentAgent:
    """
    Interactive Student Agent with step-by-step reasoning.
    
    Features:
    - Uses RAG (Pinecone) + Web Search (OpenAI) for comprehensive answers
    - Subject-aware step-by-step response format (Math, Science, History, Geography)
    - Focuses on teaching methodology, not just final answers
    - Encourages learning through guided problem-solving
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
        
        # Create dedicated tool registry
        self.tool_registry = ToolRegistry()
        self._register_tools()
        
        # Create ReAct agent
        self.react_agent = ReActAgent(
            llm=llm,
            tool_registry=self.tool_registry,
            max_iterations=real_max_iterations,
            enforce_sequential=False,
        )
    
    def _register_tools(self):
        """Register tools for interactive learning."""
        # RAG retrieval tool
        retrieval_tool = RetrievalTool(self.retriever)
        self.tool_registry.register(retrieval_tool)
        
        # Web search tool (optional)
        if self.enable_web_search:
            web_search_tool = WebSearchTool()
            self.tool_registry.register(web_search_tool)
            logger.info("--- [VERIFICATION] Interactive Student Agent: Web Search is ENABLED ---")
        else:
            logger.info("--- [VERIFICATION] Interactive Student Agent: Web Search is DISABLED ---")
        
        logger.info("Registered %d tools for Interactive Student Agent", len(self.tool_registry.list_tools()))
    
    def _build_interactive_system_prompt(self, query: str, subjects: List[str], target_lang: str, state: Optional[Dict[str, Any]] = None) -> str:
        subjects_str = ", ".join(subjects) if subjects else "General"
        
        # GRADE-BASED SOCRATIC IDENTITY
        grade = (state or {}).get("student_grade", "B")
        logger.info("--- [DEBUG] Building InteractiveAgent prompt for Grade: %s ---", grade)
        
        identities = {
            "A": {
                "name": "The Master Socratic Scout",
                "focus": "High-level technical scaffolding and critical inquiry.",
                "rules": [
                    "Assume mastery of basic premises. Focus on logical edge cases and 'why' it happens.",
                    "NEVER use numbered lists for 'Steps'. Use conversational flow.",
                    "Ask complex, multi-step scouting questions that challenge the student's mental model.",
                    "Tone: Precise, professional, and intellectually rigorous."
                ]
            },
            "B": {
                "name": "The Supportive Scout",
                "focus": "Balanced discovery, logical flow, and academic support.",
                "rules": [
                    "Break the concept into clear discovery branches.",
                    "Use standard technical terms but explain them briefly.",
                    "Ask clear, guided questions to lead the student to the next logical point.",
                    "Tone: Professional, patient, and academically helpful."
                ]
            },
            "C": {
                "name": "The Patient Guide",
                "focus": "Confidence building, simple scaffolding, and momentum.",
                "rules": [
                    "Break the concept into very easy, manageable 'mini-challenges'.",
                    "Use simple everyday language and analogies.",
                    "MANDATORY: Include a concrete example (like a ball rolling) to ground the explanation.",
                    "Ask gentle, single-step questions to build the student's confidence.",
                    "MANDATORY: Start with: 'That's a really interesting thing to think about!'"
                ]
            },
            "D": {
                "name": "The Foundational Coach",
                "focus": "Maximum empathy, micro-step discovery, and constant reassurance.",
                "rules": [
                    "Focus on one tiny 'aha!' moment at a time. No complex structures.",
                    "Use strictly basic, conversational vocabulary.",
                    "MANDATORY: Always use a simple real-world example in your scaffolding.",
                    "MANDATORY: Start and end with heavy praise (e.g., 'You're doing amazing!').",
                    "Tone: Highly enthusiastic, protective, and super simple."
                ]
            }
        }
        
        identity = identities.get(grade, identities["B"])
        identity_rules = "\n".join([f"- {r}" for r in identity["rules"]])

        # PROACTIVE EFFICIENCY RULE
        rag_quality = (state or {}).get("rag_quality", "low")
        efficiency_instruction = ""
        if rag_quality == "high":
            efficiency_instruction = "Highly relevant curriculum documents are provided. Synthesize your guidance IMMEDIATELY from these documents."
        
        prompt = f"""You are 'Vidya', acting as **{identity['name']}** for a student with Grade {grade}.
Primary Focus: {identity['focus']}

### YOUR SOCRATIC IDENTITY RULES:
{identity_rules}

### CORE OPERATIONAL RULES:
1. **NO ANSWERS**: Never just give the answer. Lead them to it.
2. **NO META-TALK**: Never say "I searched" or "Based on documents".
3. **Citations**: Use Lecture ID only.
4. **Target Language [STRICT]**: {target_lang}. The user has explicitly requested to communicate in {target_lang}. **DISREGARD** the language used in previous conversation history if it is different. Respond ENTIRELY in {target_lang}.
5. **Efficiency**: {efficiency_instruction}
6. **LOCAL KNOWLEDGE ONLY [STRICT]**: Never mention external websites, web resources, or links (e.g., "Khan Academy", "YouTube", "further reading links"). Use ONLY information from local Lecture ID documents. Web search results (if available) are for internal context ONLY and must NEVER be cited or suggested to the student.
7. **Citation Filtering [STRICT]**: Only cite and use information from documents with a **Score > 0.60**. If no documents meet this threshold, acknowledge that relevant curriculum material was not found rather than using external knowledge.

HOW TO RESPOND:
- Provide your Socratic guidance in {target_lang}, strictly embodying the **{identity['name']}** persona through the rules above.
"""
        return prompt
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process student query with interactive, step-by-step approach.
        """
        query = state["query_en"]
        history = state.get("conversation_history", [])
        
        logger.info("Interactive Student Agent processing query: %s", query[:100])
        
        # Get target language
        target_lang = state.get("language", "en")
        
        # Use pre-detected subjects from state (populated in AnalyzeQueryNode)
        subjects = state.get("subjects") or ["general"]
        subjects = [s.lower() for s in subjects]
        
        # Run ReAct reasoning loop with interactive context and target language
        self.react_agent.build_system_prompt = lambda q, s: self._build_interactive_system_prompt(q, s, target_lang, state)

        # Set correction flag for ReAct trace if validation results exist
        val_results = state.get("validation_results")
        if val_results and not val_results.get("is_valid"):
            state["is_correction"] = True
            logger.info("Retrying Interactive Agent with corrective feedback...")
        
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
            
            if result and "answer" in result:
                state["response"] = result["answer"]
                
                # Extract citations (Minimum score 0.6)
                citations = CitationService.extract_citations(result.get("reasoning_chain", []), min_score=0.6)
                if citations:
                    state["citations"] = citations
                
                state["llm_calls"] = state.get("llm_calls", 0) + result.get("iterations", 0)
                
                # Mark as translated if not English
                if target_lang != "en" and result["answer"] != FALLBACK_MESSAGE:
                    state["final_language"] = target_lang
            else:
                state["citations"] = []
            
            logger.info(
                "Interactive Student Agent completed with %d iterations",
                result.get("iterations", 0),
            )
        except Exception as exc:
            logger.error(
                "Interactive Student Agent failed for user %s, session %s: %s",
                state.get("user_id"),
                state.get("user_session_id"),
                exc,
                exc_info=True
            )
            state["response"] = FALLBACK_MESSAGE
            state["llm_calls"] = 0
            state["citations"] = []
        
        return state
    



__all__ = ["InteractiveStudentAgent"]
