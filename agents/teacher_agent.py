"""Teacher Agent with ReAct reasoning for content review and analytics."""

import logging
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from agents.react_agent import ReActAgent, FALLBACK_MESSAGE
from tools import ToolRegistry, RetrievalTool
from services.retriever import RetrieverService
from services.citation_service import CitationService
from state import AgentState, ConversationTurn
from config import settings

logger = logging.getLogger(__name__)


class TeacherAgent:
    """
    Teacher-focused agent using ReAct reasoning.
    
    Helps teachers:
    - Review session content and summaries
    - Analyze topic coverage across lectures
    - Get analytical insights about teaching
    - Identify content gaps and patterns
    
    Uses retrieval-augmented generation with analytical tone.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: RetrieverService,
        max_iterations: Optional[int] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        
        # Resolve config safely
        real_max_iterations = max_iterations or (settings.max_iterations if settings else 5)
        
        # Create dedicated tool registry for teacher analytics
        self.tool_registry = ToolRegistry()
        self._register_teacher_tools()
        
        # Create ReAct agent for teacher queries
        self.react_agent = ReActAgent(
            llm=llm,
            tool_registry=self.tool_registry,
            max_iterations=real_max_iterations,
            enforce_sequential=False,
        )
    
    def _register_teacher_tools(self):
        """Register tools for teacher analytics."""
        # Retrieval tool for searching lecture content
        retrieval_tool = RetrievalTool(self.retriever)
        self.tool_registry.register(retrieval_tool)
        
        logger.info("Registered %d tools for Teacher agent", len(self.tool_registry.list_tools()))
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process teacher query with analytical focus.
        
        The agent will:
        - Understand what the teacher wants to review
        - Search for relevant lecture content
        - Provide summaries and analytical insights
        - Group information by sessions/topics
        """
        query = state["query_en"]
        history = state.get("conversation_history", [])
        
        logger.info("Teacher Agent processing query: %s", query[:100])
        
        # Get target language
        target_lang = state.get("detected_language", state.get("language", "en"))
        
        # Run ReAct reasoning loop with teacher context and target language
        self.react_agent.build_system_prompt = lambda q, s: self._build_teacher_system_prompt(q, s, target_lang, state)
        
        # Set correction flag for ReAct trace if validation results exist
        val_results = state.get("validation_results")
        if val_results and not val_results.get("is_valid"):
            state["is_correction"] = True
            logger.info("Retrying Teacher Agent with corrective feedback...")
        
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
            
            # Extract citations once from reasoning chain
            citations = CitationService.extract_citations(result.get("reasoning_chain", []))
            
            # Update state with results
            if result and "answer" in result:
                state["response"] = result["answer"]
                state["citations"] = citations
                state["llm_calls"] = state.get("llm_calls", 0) + result.get("iterations", 0)
                
                # If the agent generated the response in the target language (and it's not the fallback),
                # mark it as translated so we skip the translation node.
                if target_lang != "en" and result["answer"] != FALLBACK_MESSAGE:
                    state["is_translated"] = True
                    state["final_language"] = target_lang
            else:
                citations = []
            
            logger.info(
                "Teacher Agent completed with %d iterations, %d citations",
                result.get("iterations", 0),
                len(citations),
            )
        except Exception as exc:
            logger.error(
                "Teacher Agent execution failed for user %s, session %s, query: %s - Error: %s",
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
    

    def _build_teacher_system_prompt(
        self,
        query: str,
        subjects: List[str],
        target_lang: str = "en",
        state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build teacher-specific system prompt."""
        subjects_str = ", ".join(subjects) if subjects else "General"
        
        # PROACTIVE EFFICIENCY RULE
        rag_quality = (state or {}).get("rag_quality", "low")
        efficiency_instruction = ""
        if rag_quality == "high":
            efficiency_instruction = "\n- **EFFICIENCY RULE**: Highly relevant curriculum documents are already provided in your context. Answer IMMEDIATELY and DIRECTLY using these documents. Do NOT call 'retrieve_documents' again unless they are insufficient."
        elif rag_quality == "medium":
            efficiency_instruction = "\n- **EFFICIENCY RULE**: Good curriculum documents are available in context. Use them as your primary source."

        # CORRECTION FEEDBACK
        correction_instruction = ""
        val_results = (state or {}).get("validation_results")
        if val_results and not val_results.get("is_valid"):
            feedback = val_results.get("feedback")
            correction_instruction = f"\n\n> [!IMPORTANT]\n> **PREVIOUS ATTEMPT FAILED VALIDATION**:\n> {feedback}\n> Please correct your response based on the feedback."

        # Get available tools
        tools_text = self.tool_registry.format_for_prompt()
        
        prompt = f"""You are 'Vidya', an expert Teacher and Content Creator. 
Your goal is to provide deep academic insights and professional guidance.

- **Target Language**: {target_lang} (Respond ONLY in this language)
- **Detected Subjects**: {subjects_str}{efficiency_instruction}
- **Persona**: Authoritative, scholarly, and pedagogical.{correction_instruction}

Available Tools:
{tools_text}

CRITICAL RULES FOR TEACHER ASSISTANCE:

1. **RAG-ONLY**: Answer ONLY from retrieved lecture content. NO general knowledge.
   - **DIRECT ANSWERS ONLY**: Provide analysis directly. NEVER mention the retrieval process.
   - **SILENT FAILURE**: If no documents are found, NEVER say "no information found". Instead, ask a professional clarifying question about the analytical goal.
   - **AMBIGUITY HANDLING**: If retrieval returns mixed results, DO NOT proceed. List detected contexts and ask for the "main objective".
   - **BREVITY (MANDATORY)**: Keep your response concise (50-100 tokens). Unless the user asks for more detail, provide only core analysis.
   - If the query is ambiguous, ask clarifying questions.

3. **ANALYTICAL TONE**: 
   - Use professional, analytical language
   - Provide summaries, not detailed explanations
   - Group information by sessions/topics
   - Identify patterns and gaps

4. **TEACHER-FOCUSED RESPONSES**:
   - For broad queries (e.g., "what did I teach?"): Provide "Coverage Analysis" or "Session Summary"
   - For specific questions (e.g., "what is X?"): Provide a direct, factual answer based on your content
   - "You covered X in session Y"
   - "Topics taught: 1. A, 2. B, 3. C"

5. **QUERY UNDERSTANDING**:
   - "What did I teach in session X?" → Retrieve session X content, summarize
   - "Show me all DevOps topics" → Search DevOps, group by sessions
   - "What's in Chapter Y?" → Retrieve chapter content, list topics
   - "Which brain structure...?" → Retrieve specific details, answer directly

6. **RESPONSE FORMAT**:
   - Start with summary/overview
   - Provide structured information
   - DO NOT list specific session details in the text (these go in citations)
   - Focus on the *content* and *analysis*
   - **LANGUAGE**: Your final response MUST be in {target_lang}.

HOW TO RESPOND:

- **TO SEARCH**: Call the `retrieve_documents` tool. Do not write "Action:" or "Thought:" in your text response, just use the tool.
- **TO ANSWER**: Write your final response directly in {target_lang}.

EXAMPLES:

Example 1 - Session Summary:
(Call Tool: retrieve_documents("session 10 content topics"))

Example 2 - Topic Coverage:
(Call Tool: retrieve_documents("DevOps CI/CD Docker Kubernetes"))

Example 3 - Analytical Response (Coverage) (User Language: English):
(After retrieval)
DevOps Coverage Analysis:

DevOps concepts were covered across multiple sessions, focusing on CI/CD basics, Docker containerization, and Kubernetes orchestration. The curriculum emphasizes the practical application of these tools in automating deployment pipelines.

Total: 3 sessions, 7 topics covered
Key concepts: CI/CD, Docker, Kubernetes, Automation

[Citations: session_10, session_12, session_15]

Example 4 - Specific Question:
(After retrieval)
The amygdala is the brain structure mainly involved in processing fear and emotional responses. It is an almond-shaped structure sitting next to the hippocampus.

[Citation: Lecture ID 1]
"""
        return prompt
    



__all__ = ["TeacherAgent"]
