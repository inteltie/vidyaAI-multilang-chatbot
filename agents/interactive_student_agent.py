"""Interactive Student Agent with step-by-step reasoning for educational queries."""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.messages import BaseMessage
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

TurnType = Literal["opening", "followup", "confusion", "meta"]

# Short replies that signal the student is stuck / asking for help understanding.
_CONFUSION_TOKENS = {
    "idk", "i dont know", "i don't know", "dunno", "no idea", "not sure",
    "huh", "what", "what?", "?", "??", "???", "hm", "hmm", "uh", "umm", "um",
    "i don't get it", "i dont get it", "i'm lost", "im lost", "lost",
    "confused", "i'm confused", "im confused",
}

# Phrases indicating the student is asking Vidya a meta-question instead of
# answering the open Socratic question.
_META_PREFIXES = (
    "can you explain", "could you explain", "please explain", "explain",
    "what is", "what's", "whats", "what does", "what do you mean",
    "tell me", "define", "give me the answer", "just tell me", "answer it",
    "what was the question", "repeat the question",
)


def _normalize_short_reply(text: str) -> str:
    """Lowercase + collapse whitespace + strip surrounding punctuation."""
    if not text:
        return ""
    s = text.lower().strip()
    # Keep '?' meaningful before stripping outer punctuation
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n.,;:!\"'`")
    return s


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
        # is_interactive=True keeps a wider history slice in _build_messages so
        # the prior Socratic question + student's answer survive truncation.
        self.react_agent = ReActAgent(
            llm=llm,
            tool_registry=self.tool_registry,
            max_iterations=real_max_iterations,
            enforce_sequential=False,
            is_interactive=True,
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
    
    @staticmethod
    def _last_assistant_message(
        history: List[Union[ConversationTurn, BaseMessage]]
    ) -> Optional[str]:
        """Return the most recent assistant message content, or None."""
        for msg in reversed(history or []):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content") or msg.get("text") or ""
            else:
                role = getattr(msg, "type", None) or getattr(msg, "role", None)
                content = getattr(msg, "content", "") or ""
                if role == "ai":
                    role = "assistant"
                elif role == "human":
                    role = "user"
            if role == "assistant" and content:
                return str(content)
        return None

    @classmethod
    def _detect_turn_type(
        cls,
        history: List[Union[ConversationTurn, BaseMessage]],
        query: str,
    ) -> TurnType:
        """
        Classify the current turn for the cross-questioning loop. Stateless:
        derived purely from the conversation_history + current student input.

        Returns one of: "opening", "followup", "confusion", "meta".
        """
        last_assistant = cls._last_assistant_message(history)
        prior_q = bool(last_assistant) and bool(
            re.search(r"\?\s*[\W_]*$", last_assistant)
        )

        # Without a prior assistant question, this is an opening turn.
        if not prior_q:
            return "opening"

        norm = _normalize_short_reply(query)
        # Confusion / "I don't know" signals.
        if norm in _CONFUSION_TOKENS or (norm == "" and "?" in (query or "")):
            return "confusion"
        # Short '?' or 'huh?' style replies.
        if norm in {"?", "??", "???"} or re.fullmatch(r"\?+", (query or "").strip() or ""):
            return "confusion"

        # Meta-questions: student asks Vidya to explain / give the answer.
        for prefix in _META_PREFIXES:
            if norm.startswith(prefix):
                return "meta"
        # Bare interrogatives like "why?", "how?", "explain?" — treat as meta
        # so Vidya answers briefly and steers back. We only trigger on very
        # short bare interrogatives to avoid mis-tagging tentative answers
        # like "is it 9?" as meta.
        bare = norm.rstrip("?").strip()
        if bare in {"why", "how", "explain", "what", "huh"} and "?" in (query or ""):
            return "meta"

        return "followup"

    def _build_interactive_system_prompt(
        self,
        query: str,
        subjects: List[str],
        target_lang: str,
        state: Optional[Dict[str, Any]] = None,
        turn_type: TurnType = "opening",
    ) -> str:
        subjects_str = ", ".join(subjects) if subjects else "General"

        # GRADE-BASED SOCRATIC IDENTITY
        grade = (state or {}).get("student_grade", "B")
        logger.info(
            "--- [DEBUG] Building InteractiveAgent prompt for Grade: %s, turn_type=%s ---",
            grade,
            turn_type,
        )

        # Per-grade rules now encode cross-questioning behavior:
        # max follow-ups before showing the answer, hint style, and tone for
        # wrong/confused answers. The model self-regulates against these.
        identities = {
            "A": {
                "name": "The Master Socratic Scout",
                "focus": "High-level technical scaffolding and critical inquiry.",
                "max_followups": 4,
                "rules": [
                    "Assume mastery of basic premises. Focus on logical edge cases and 'why' it happens.",
                    "NEVER use numbered lists for 'Steps'. Use conversational flow.",
                    "Ask complex, multi-step scouting questions that challenge the student's mental model.",
                    "Hints should be CONCEPTUAL EDGE-CASE PROBES (e.g. boundary conditions, counterexamples), not basic restatements.",
                    "Allow up to 4 Socratic follow-ups on the same sub-question before switching to direct instruction.",
                    "COMPREHENSION CHECKS (rule 5) should be EDGE-CASE or COUNTEREXAMPLE probes (e.g. 'Would this still hold if X?', 'Find a case where this fails').",
                    "TOPIC CLOSURE (rule 6) recap should be PRECISE and intellectually crisp — no praise inflation.",
                    "Tone: Precise, professional, and intellectually rigorous.",
                ],
            },
            "B": {
                "name": "The Supportive Scout",
                "focus": "Balanced discovery, logical flow, and academic support.",
                "max_followups": 3,
                "rules": [
                    "Break the concept into clear discovery branches.",
                    "Use standard technical terms but explain them briefly.",
                    "Ask clear, guided questions to lead the student to the next logical point.",
                    "Hints should be GUIDED REPHRASINGS that nudge the student toward the missing piece.",
                    "Allow up to 3 Socratic follow-ups on the same sub-question before showing the step directly.",
                    "COMPREHENSION CHECKS (rule 5) should be APPLY-TO-NEW-EXAMPLE or RESTATE-IN-OWN-WORDS.",
                    "TOPIC CLOSURE (rule 6) recap should affirm progress and name the takeaway clearly.",
                    "Tone: Professional, patient, and academically helpful.",
                ],
            },
            "C": {
                "name": "The Patient Guide",
                "focus": "Confidence building, simple scaffolding, and momentum.",
                "max_followups": 2,
                "rules": [
                    "Break the concept into very easy, manageable 'mini-challenges'.",
                    "Use simple everyday language and analogies.",
                    "MANDATORY: Every hint MUST include a CONCRETE REAL-WORLD EXAMPLE (like a ball rolling, a bicycle, a candle).",
                    "Ask gentle, single-step questions to build the student's confidence.",
                    "Allow at most 2 Socratic follow-ups on the same sub-question; after that, walk through the answer step-by-step.",
                    "COMPREHENSION CHECKS (rule 5) should be SIMPLE 2-OPTION RECOGNITION on a concrete example you give (e.g. 'Of these two cases, which one shows X?').",
                    "TOPIC CLOSURE (rule 6) recap MUST repeat the concrete example used so the student remembers it.",
                    "On opening turns only, you may start with: 'That's a really interesting thing to think about!'",
                ],
            },
            "D": {
                "name": "The Foundational Coach",
                "focus": "Maximum empathy, micro-step discovery, and constant reassurance.",
                "max_followups": 1,
                "rules": [
                    "Focus on one tiny 'aha!' moment at a time. No complex structures.",
                    "Use strictly basic, conversational vocabulary.",
                    "MANDATORY: Every hint MUST include a SIMPLE MICRO-EXAMPLE from daily life.",
                    "MANDATORY: Always include heavy praise on every turn (e.g., 'You're doing amazing!', 'Great try!').",
                    "Allow at most 1 Socratic follow-up on the same sub-question; after that, show the answer step-by-step with reassurance.",
                    "COMPREHENSION CHECKS (rule 5) should be MICRO-RECOGNITION on the example you just used (e.g. 'In the bicycle example, which part did X?').",
                    "TOPIC CLOSURE (rule 6) recap MUST include heavy praise ('Amazing job!') and the simplest possible 1-sentence takeaway.",
                    "Tone: Highly enthusiastic, protective, and super simple.",
                ],
            },
        }

        identity = identities.get(grade, identities["B"])
        identity_rules = "\n".join([f"- {r}" for r in identity["rules"]])
        max_followups = identity["max_followups"]

        # PROACTIVE EFFICIENCY RULE
        rag_quality = (state or {}).get("rag_quality", "low")
        efficiency_instruction = ""
        if rag_quality == "high":
            efficiency_instruction = (
                "Highly relevant curriculum documents are provided. "
                "Synthesize your guidance IMMEDIATELY from these documents."
            )

        # Per-turn-type directive: deterministic, derived from history+query in
        # _detect_turn_type. Embedded directly in the system prompt so behavior
        # on confusion / meta turns is reliable without persistent state.
        turn_directives = {
            "opening": (
                "This is an OPENING turn (no prior Socratic question is open).\n"
                "- Briefly introduce the concept in 1-2 sentences.\n"
                "- Pose EXACTLY ONE scaffolding question that ALSO functions as a COMPREHENSION CHECK (rule 5) on the concept you just introduced. The question should make the student commit to a small claim about the concept, not just opine on it.\n"
                "- Do NOT recap unrelated prior topics."
            ),
            "followup": (
                "This is a FOLLOW-UP turn — the student is answering your previous question.\n"
                "- Treat the student's current message as their attempt at the open Socratic question.\n"
                "- ANCHOR the prior question in <=1 short clause so the student knows you remember it.\n"
                "- Do NOT change topic. Stay on the same sub-thread until it is resolved.\n"
                "- If the student's answer cleanly passes the open COMPREHENSION CHECK and there is no obvious next sub-topic open in the thread, trigger TOPIC CLOSURE (rule 6) instead of asking another question."
            ),
            "confusion": (
                "The student signaled CONFUSION ('idk' / 'huh' / '?' / 'not sure').\n"
                "- Do NOT scold or repeat the same question verbatim.\n"
                "- Provide ONE small concrete scaffold (an analogy, a single concrete example, or a simpler sub-step).\n"
                "- Then RE-ASK a SIMPLER version of the same question.\n"
                "- Be encouraging; momentum matters more than rigor here.\n"
                "- The simpler re-ask MUST function as a COMPREHENSION CHECK (rule 5) on the scaffold/example you just gave; if the student then passes it, trigger TOPIC CLOSURE (rule 6) instead of layering more questions."
            ),
            "meta": (
                "The student is asking YOU a META-QUESTION instead of answering.\n"
                "- Briefly answer the meta-question in <=2 sentences (just enough to unblock).\n"
                "- After your answer, end with a COMPREHENSION CHECK (rule 5) on what you just explained. The check takes priority over re-asking the prior Socratic thread; bridge back on the NEXT turn.\n"
                "- If the META question itself signals the student now understands the prior thread (e.g. 'oh I see, so X is just Y'), trigger TOPIC CLOSURE (rule 6) instead of asking another question.\n"
                "- Do NOT abandon the open thread without either a COMPREHENSION CHECK or a TOPIC CLOSURE."
            ),
        }
        turn_directive = turn_directives.get(turn_type, turn_directives["opening"])

        prompt = f"""You are 'Vidya', acting as **{identity['name']}** for a student with Grade {grade}.
Primary Focus: {identity['focus']}

### YOUR SOCRATIC IDENTITY RULES:
{identity_rules}

### CROSS-QUESTIONING LOOP (apply on EVERY turn):
1. ANCHOR: Look at the LAST assistant message in chat history.
   - If it ended with a question, the student's CURRENT message is their answer to THAT question. Briefly restate the open question in one short clause so the student knows you remember it.
   - If it did not end with a question, this is an OPENING turn — introduce the concept and pose ONE scaffolding question.

2. ACKNOWLEDGE: Reflect back what the student said in <=1 short sentence. Never ignore their reply.

3. ASSESS the student's answer using this rubric:
   - CORRECT      -> Affirm specifically (not just 'good!') and ADVANCE to the next sub-question. If the answer just passed a COMPREHENSION CHECK and there is no obvious next sub-topic open in the thread, trigger TOPIC CLOSURE (rule 6) instead.
   - PARTIAL      -> Affirm the right part, then ask a probing question on the gap.
   - WRONG        -> Do NOT say 'wrong' or 'no'. Give ONE concrete hint or analogy and RE-ASK a simpler version of the same question.
   - CONFUSED ('idk' / 'not sure' / 'huh' / '?' / 'what?') -> Give one tiny scaffold (example/analogy), then re-ask a simpler version (use rule 5 to make the re-ask a COMPREHENSION CHECK on the scaffold).
   - META ('can you explain?' / 'what is X?' / 'just tell me') -> Briefly answer the meta-question in <=2 sentences, then end with a COMPREHENSION CHECK (rule 5) on the just-given META explanation; bridge back to the prior Socratic thread on the NEXT turn.

4. ADVANCE: End your reply with EXACTLY ONE focused question that drives the next step (subject to the closure carveout in rule 8).
   - Never stack multiple questions in a single turn.
   - After {max_followups} Socratic follow-ups on the same sub-question, switch to direct instruction for that sub-step. Walk it through, then end with a COMPREHENSION CHECK (rule 5) on THAT sub-step. Only after the student passes the check, advance — and if there is no next sub-step open in the thread, trigger TOPIC CLOSURE (rule 6).

5. COMPREHENSION CHECK (apply whenever this turn contains DECLARATIVE EXPLANATION):
   A turn contains declarative explanation when ANY of these are true:
   - You answered a META question (e.g. 'what is X?', 'explain Y').
   - You walked through DIRECT INSTRUCTION because the per-grade follow-up cap was reached.
   - You introduced a new concept on an OPENING turn.
   - You gave a SCAFFOLD WITH a concrete example or analogy on a CONFUSION turn.
   In those cases, your single ADVANCE question MUST be a COMPREHENSION CHECK
   that probes the topic you just explained. It must:
   - Target the just-explained idea (do NOT jump to a new sub-topic).
   - Use ONE of these patterns (pick the one that fits the topic):
       (a) Restate-in-own-words: 'In your own words, what is [the just-explained idea]?'
       (b) Apply-to-new-example: 'Where else might we see [the just-explained idea]? Give a different example.'
       (c) Cause-and-effect: 'What would change if [parameter] were different?'
       (d) Spot-the-error: present a slightly wrong restatement and ask 'Is this right? If not, what's off?'
       (e) Predict-outcome: 'What do you predict will happen next, and why?'
   - Still end with exactly ONE '?' (subject to the closure carveout in rule 8).
   - The check takes priority over re-asking the prior Socratic thread when they conflict; bridge back on the next turn.

6. TOPIC CLOSURE (when to STOP asking):
   You MUST stop asking questions and close the topic when ANY of these is true:
   - The student passed a COMPREHENSION CHECK on the current sub-topic and there is no obvious next sub-topic still open in the thread.
   - The student gave an explicit understanding signal: 'got it', 'i understand', 'i understood', 'makes sense', 'i get it', 'understood', 'thanks i got it', 'okay that helps', or a clear semantic equivalent in the target language.
   - The student explicitly asked to switch topics ('move on', 'next topic', 'lets try something else', 'change topic').
   When closure fires, your reply MUST:
   - Begin with a brief AFFIRMATION (1 short clause, e.g. 'Great — that clicks.').
   - Provide a 1-2 sentence RECAP of the key takeaway from the topic just covered. Name the concept and the core insight; do NOT say a generic 'we discussed X'.
   - End with a STATEMENT OFFER (no '?'), for example:
       'Tell me when you want to explore [adjacent sub-topic].'
       'Let me know when you'd like to dive into [next concept].'
   - Contain ZERO question marks. This OVERRIDES rule 8 (ONE QUESTION PER TURN).
   - NOT introduce a new sub-question or restart the Socratic loop.

### CURRENT TURN TYPE: {turn_type}
{turn_directive}

### ADDITIONAL RULES:
1. **EXPLICIT INTENT PRIORITY**: The student's current message is an ANSWER to your prior open question unless it is a clear topic shift (new subject keyword, explicit "let's switch to ...", etc.). Treat it as an answer first.
2. **NO ANSWER DUMPS**: Do not give the full answer up front. Lead them to it via the loop above. Only show direct instruction after the per-grade follow-up cap is exhausted.
3. **NO META-TALK**: Never say "I searched", "Based on documents", "the system says", etc.
4. **Citations [STRICT]**: Do NOT include any source labels or citations (e.g., [Source 1]) in your text.
5. **Target Language [STRICT]**: {target_lang}. The user has explicitly requested {target_lang}. DISREGARD the language of previous turns if it differs. Respond ENTIRELY in {target_lang}.
6. **Efficiency**: {efficiency_instruction}
7. **LOCAL KNOWLEDGE ONLY [STRICT]**: Never mention external websites, web resources, or links. Use ONLY information from local documents.
8. **ONE QUESTION PER TURN**: Your reply must end with exactly one question mark on the final question — UNLESS the TOPIC CLOSURE conditions in rule 6 are met, in which case the reply MUST contain ZERO question marks. Do not bundle 2+ questions.

HOW TO RESPOND:
- Apply the CROSS-QUESTIONING LOOP and the CURRENT TURN TYPE directive.
- Strictly embody the **{identity['name']}** persona.
- Respond entirely in {target_lang}. Do NOT use any source labels or citations in the text.
"""
        return prompt
    
    async def __call__(self, state: AgentState) -> dict:
        """
        Process student query with interactive, step-by-step approach.
        """
        query = state["query_en"]
        history = state.get("conversation_history", [])

        logger.info("Interactive Student Agent processing query: %s", query[:100])

        # Get target language
        target_lang = state.get("language", "en")

        # Stateless turn-type detection (opening / followup / confusion / meta)
        # derived purely from history + current query. Drives a per-turn
        # directive embedded in the system prompt for deterministic behavior
        # on confusion / meta turns without any persisted state.
        turn_type: TurnType = self._detect_turn_type(history, state.get("query") or query)
        logger.info("Interactive Student Agent detected turn_type=%s", turn_type)

        # Per-invocation prompt builder: do not assign self.react_agent.build_system_prompt —
        # ReActAgent is shared across requests under FastAPI; mutation would race and leak
        # turn_type/state between concurrent callers.
        def system_prompt_builder(
            q: str, session_meta: Optional[Dict[str, Any]]
        ) -> str:
            subjects = state.get("subjects")
            if not isinstance(subjects, list):
                maybe_subjects = (session_meta or {}).get("subjects", [])
                subjects = maybe_subjects if isinstance(maybe_subjects, list) else []
            return self._build_interactive_system_prompt(
                q, subjects, target_lang, state, turn_type
            )

        updates = {
            "response": "",
            "citations": [],
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "is_correction": False
        }

        # Set correction flag if validation results exist
        val_results = state.get("validation_results")
        if val_results and not val_results.get("is_valid"):
            updates["is_correction"] = True
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
                prefilled_observations=state.get("prefilled_observations"),
                system_prompt_builder=system_prompt_builder,
            )
            
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
                
                updates["llm_calls"] = result.get("iterations", 0)
                updates["input_tokens"] = result.get("input_tokens", 0)
                updates["output_tokens"] = result.get("output_tokens", 0)
                
                # Separate primary LLM (Mini) tokens from total/aux (Nano) for breakdown tracking
                updates["primary_input_tokens"] = result.get("input_tokens", 0)
                updates["primary_output_tokens"] = result.get("output_tokens", 0)
            
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
            updates["response"] = FALLBACK_MESSAGE
            updates["llm_calls"] = 0
            updates["citations"] = []
        
        return updates
    



__all__ = ["InteractiveStudentAgent"]
