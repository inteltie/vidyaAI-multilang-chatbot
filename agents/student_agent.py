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
        max_iterations: int = settings.max_iterations,
        enable_web_search: bool = True,
    ):
        self.llm = llm
        self.retriever = retriever
        self.enable_web_search = enable_web_search
        
        # Create dedicated tool registry for student learning
        self.tool_registry = ToolRegistry()
        self._register_student_tools()
        
        # Create ReAct agent for student learning
        self.react_agent = ReActAgent(
            llm=llm,
            tool_registry=self.tool_registry,
            max_iterations=max_iterations,
            enforce_sequential=False,
        )
    
    def _build_student_system_prompt(self, query: str, subjects: List[str], target_lang: str, state: Optional[Dict[str, Any]] = None) -> str:
        subjects_str = ", ".join(subjects) if subjects else "General"
        
        # PROACTIVE EFFICIENCY RULE
        rag_quality = (state or {}).get("rag_quality", "low")
        efficiency_instruction = ""
        if rag_quality == "high":
            efficiency_instruction = "\n- **EFFICIENCY RULE**: Highly relevant curriculum documents are already provided in your context. Answer IMMEDIATELY and DIRECTLY using these documents. Only use web search if they do not contain the answer. Do NOT call 'retrieve_documents' again."
        elif rag_quality == "medium":
            efficiency_instruction = "\n- **EFFICIENCY RULE**: Good curriculum documents are available in context. Use them as your primary source."
        
        # CORRECTION FEEDBACK
        correction_instruction = ""
        val_results = (state or {}).get("validation_results")
        if val_results and not val_results.get("is_valid"):
            feedback = val_results.get("feedback")
            correction_instruction = f"\n\n> [!IMPORTANT]\n> **PREVIOUS ATTEMPT FAILED VALIDATION**:\n> {feedback}\n> Please correct your explanation based on the feedback and documents provided."

        prompt = f"""You are 'Vidya', a **Fact-First Synthesizer**.
Your goal is to provide a comprehensive, direct, and technical overview by synthesizing official curriculum documents and web-based enrichment.

- **Target Language**: {target_lang} (Respond ONLY in this language)
- **Detected Subjects**: {subjects_str}{efficiency_instruction}
- **Persona**: Professional, brief, and information-dense. Avoid all pedagogical "fluff" or conversational filler.

### INSTRUCTIONS:
1. Use `retrieve_documents` to find curriculum content.
2. Only use `web_search` as a LAST RESORT if the retrieved documents are completely insufficient.
3. Provide citations using Lecture ID only (no chunk references).
4. If the query is ambiguous (e.g., "Transformers"), check the documents. If they mention multiple contexts, ask the user for their "main objective".{correction_instruction}
"""
        # Get available tools
        tools_text = self.tool_registry.format_for_prompt()
        
        prompt += f"""
Available Tools:
{tools_text}

Student Query: {query}

=== YOUR OPERATIONAL STRATEGY ===

1. **DUAL-SOURCE SYNTHESIS**: 
   - You SHOULD call `retrieve_documents` and `web_search` in PARALLEL in the same turn if the query benefits from both curriculum data and external enrichment (e.g., "Explain X based on my lectures but also give me a real-world example from the web").
   - **PRIORITY**: Always prioritize `retrieve_documents` for curriculum grounding. Use `web_search` to fill gaps, provide recent examples, or as a fallback if the curriculum is silent.
   - Integrate information from both sources into a single, cohesive response.

2. **CONCISE OVERVIEW PERSONA**:
   - Provide a high-level summary/overview of the solution.
   - Be direct, factual, and time-efficient.
   - **DO NOT** use pedagogical step-by-step walkthroughs (like "Step 1: Understand").
   - **DO NOT** use Socratic hints or "Check for Understanding" questions at the end.

=== CRITICAL RULES ===

1. **DIRECT ANSWERS ONLY**: Provide the answer/overview directly. NEVER mention the retrieval process (e.g., "I searched", "According to the documents").
2. **SILENT FAILURE**: If no information is found in both RAG and Web, ask a proactive clarifying question about the concept.
3. **AMBIGUITY HANDLING**: If the retrieved results cover multiple distinct topics or domains (e.g., 'Transformers' in an Electrical context vs. AI Neural Networks vs. Matrix Transformations), **DO NOT** provide a full answer yet. Briefly describe the options found and ask for the student's **"main objective"** to narrow it down.
4. **SOURCE ATTRIBUTION**: If asked about lecture details, provide Lecture ID and Chapter from the metadata.
5. **LANGUAGE**: Your final response MUST be in {target_lang}.

HOW TO RESPOND:
- **TO SEARCH**: Use tools.
- **TO ANSWER**: Provide the concise overview directly in {target_lang}.
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
                citations = CitationService.extract_citations(result.get("reasoning_chain", []))
                if citations:
                    state["citations"] = citations
                
                state["llm_calls"] = result.get("iterations", 0)
                
                # If the agent generated the response in the target language (and it's not the fallback),
                # mark it as translated so we skip the translation node.
                if target_lang != "en" and result["answer"] != FALLBACK_MESSAGE:
                    state["is_translated"] = True
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
