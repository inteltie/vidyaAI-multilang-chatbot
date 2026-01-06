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


# Subject-specific step-by-step templates
STEP_BY_STEP_TEMPLATES = {
    "math": """
**MATH Problem-Solving Approach:**
1. **Understand**: Identify what is given and what is asked.
2. **Plan**: State the formulas/concepts needed.
3. **Solve**: Show each calculation step with clear explanation.
4. **Verify**: Check that the answer makes sense.
""",
    "science": """
**SCIENCE Explanation Approach:**
1. **Observe**: State the phenomenon or concept being discussed.
2. **Explain**: Break down the underlying principle step-by-step.
3. **Relate**: Connect to real-world examples the student can understand.
4. **Summarize**: Provide a concise takeaway.
""",
    "history": """
**HISTORY Analysis Approach:**
1. **Context**: Provide the historical context (time, place, circumstances).
2. **Key Facts**: List important dates, names, and places.
3. **Significance**: Explain why this event/person matters.
4. **Connections**: Link to related events or modern relevance.
""",
    "geography": """
**GEOGRAPHY Exploration Approach:**
1. **Location**: Describe where it is and its key features.
2. **Characteristics**: Explain physical and human geography aspects.
3. **Connections**: Relate to climate, economy, or culture.
4. **Significance**: Explain its importance in the broader world.
""",
    "default": """
**Learning Approach:**
1. **Identify**: Understand the core question or concept.
2. **Explore**: Break down the topic into understandable parts.
3. **Explain**: Provide clear explanations with examples.
4. **Summarize**: Give a concise takeaway.
"""
}


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
        max_iterations: int = settings.max_iterations,
        enable_web_search: bool = True,
    ):
        self.llm = llm
        self.retriever = retriever
        self.enable_web_search = enable_web_search
        
        # Create dedicated tool registry
        self.tool_registry = ToolRegistry()
        self._register_tools()
        
        # Create ReAct agent
        self.react_agent = ReActAgent(
            llm=llm,
            tool_registry=self.tool_registry,
            max_iterations=max_iterations,
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
        
        # PROACTIVE EFFICIENCY RULE
        rag_quality = (state or {}).get("rag_quality", "low")
        efficiency_instruction = ""
        if rag_quality == "high":
            efficiency_instruction = "\n- **EFFICIENCY RULE**: Highly relevant curriculum documents are already provided in your context. Use them IMMEDIATELY to guide the student. Do NOT call 'retrieve_documents' again unless they are insufficient."
        elif rag_quality == "medium":
            efficiency_instruction = "\n- **EFFICIENCY RULE**: Good curriculum documents are available in context. Use them as your primary source."
            
        # CORRECTION FEEDBACK
        correction_instruction = ""
        val_results = (state or {}).get("validation_results")
        if val_results and not val_results.get("is_valid"):
            feedback = val_results.get("feedback")
            correction_instruction = f"\n\n> [!IMPORTANT]\n> **PREVIOUS ATTEMPT FAILED VALIDATION**:\n> {feedback}\n> Please correct your explanation based on the feedback and documents provided."

        prompt = f"""You are 'Vidya', a **Socratic Tutor** and Wise Learning Guide. 
Your goal is to lead the student toward a deep understanding through storytelling, analogies, and interactive scaffolding.

- **Target Language**: {target_lang} (Respond ONLY in this language)
- **Detected Subjects**: {subjects_str}{efficiency_instruction}
- **Persona**: Patient, encouraging, and metaphorical. focus on the *process* of learning, not just the answer.{correction_instruction}
"""
        
        # Select best matching template
        primary_subject = "default"
        if subjects and len(subjects) > 0:
            # Check if any detected subject maps to a known template
            for subj in subjects:
                if subj in STEP_BY_STEP_TEMPLATES and subj != "default":
                    primary_subject = subj
                    break
        
        step_template = STEP_BY_STEP_TEMPLATES.get(primary_subject, STEP_BY_STEP_TEMPLATES["default"])
        logger.info("Selected interactive template: %s for subjects: %s", primary_subject, subjects)
        
        # Get available tools
        tools_text = self.tool_registry.format_for_prompt()
        
        prompt += f"""
Available Tools:
{tools_text}

=== YOUR TEACHING METHODOLOGY ===

{step_template}

=== CRITICAL RULES ===

1. **DIRECT ANSWERS ONLY**: Guide the student directly. NEVER mention the retrieval or search process.
2. **SILENT FAILURE**: If no documents are found, NEVER admit it. Instead, bridge to a related concept or ask a proactive scaffolding question to guide the student's thinking.
3. **AMBIGUITY HANDLING**: If the search returns documents covering multiple distinct topics, **STOP** before providing a comprehensive explanation. Ask the student for their **"main objective"**.
4. **PROACTIVE SCAFFOLDING**: Do NOT just answer with a question. Scaffold the learning: "That's a great question! To understand X, we first need to look at Y..."
5. **USE BOTH SOURCES (PARALLEL)**: You SHOULD call `retrieve_documents` and `web_search` in PARALLEL in the same turn to provide a comprehensive and fast response.
6. **TONE :: THE "WISE GUIDE"**: Be encouraging: "You're on the right track!", "This is a tricky concept, let's break it down."
   - Be patient and clear.
   - Avoid sounding like a strict examiner.

4. **CLARITY & CONTEXT**:
   - If the query is vague, ASK for more details, but offer a probable path forward.
   - "Are you asking about X or Y? I can explain X if you like."

5. **SHOW THE PROCESS**:
   - **Math**: Show calculation steps clearly (MathJax).
   - **Science**: Explain the mechanism (How/Why).
   - **Humanities**: Connect facts to the broader narrative.

6. **RESPONSE STRUCTURE (MANDATORY)**:
   - **Encouraging Opener**: Acknowledge the question positively.
   - **Step-by-Step Explanation**: Use the methodology above.
   - **Check for Understanding**: **ALWAYS** end with a specific, applied question. 
     - *Bad*: "Do you understand?"
     - *Good*: "Now that we've seen how this works, what would happen if we doubled the force?"
   - **Language**: Respond entirely in {target_lang}.

=== EXAMPLE (Math) ===

Student: "Solve 2x + 5 = 15"

Response:
"That's a classic algebra problem! We can solve this by peeling away the layers to find x.

**Step 1: Understand**
We want to get 'x' all by itself on one side of the equals sign.

**Step 2: Plan & Solve**
1. First, let's move the constant (+5) to the other side.
   2x + 5 - 5 = 15 - 5
   2x = 10
   
   (See how we did the opposite of adding 5?)

2. Now, we have 2 times x. To undo multiplication, we divide.
   2x / 2 = 10 / 2
   x = 5

**Step 3: Check**
If we put 5 back in: 2(5) + 5 is 10 + 5, which is indeed 15!

**Check for Understanding**
If the equation was 3x + 5 = 20, what would be your first step?"

=== CONTEXT FOR THIS TURN ===
User Language: {target_lang}
Detected Subjects: {', '.join(subjects)}
Student Query: {query}

=== NOW HELP THE STUDENT ===
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
                
                # Extract citations
                citations = CitationService.extract_citations(result.get("reasoning_chain", []))
                if citations:
                    state["citations"] = citations
                
                state["llm_calls"] = result.get("iterations", 0)
                
                # Mark as translated if not English
                if target_lang != "en" and result["answer"] != FALLBACK_MESSAGE:
                    state["is_translated"] = True
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
