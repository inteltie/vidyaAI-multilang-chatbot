
import asyncio
from unittest.mock import MagicMock, AsyncMock
from state import AgentState
from agents.conversational_agent import ConversationalAgent
from agents.student_agent import StudentAgent

async def test_conversational_persona_logic():
    print("Testing ConversationalAgent Persona Logic...")
    
    # Setup Mock LLM
    llm_mock = AsyncMock()
    resp_mock = MagicMock()
    resp_mock.content = "How can I help you today?"
    llm_mock.ainvoke.return_value = resp_mock
    
    agent = ConversationalAgent(llm_mock)
    
    # Case A: Generic Greeting "Hi"
    state = {
        "query": "Hi",
        "conversation_history": [],
        "session_metadata": {"summary": "Old context about Physics"},
        "is_session_restart": False,
        "llm_calls": 0
    }
    await agent(state)
    
    # Verify prompt contains the "How can I help" instruction
    prompt = llm_mock.ainvoke.call_args[0][0]
    # NOTE: ConversationalAgent has some template-based logic that might bypass LLM for "hi"
    # Actually, "hi" is not in the template list (thanks, bye, etc ARE).
    
    print("\n--- Prompt for 'Hi' ---")
    print(prompt)
    assert "DO NOT recap previous topics" in prompt
    assert "How can I help you today?" in prompt
    
    # Case B: Session Restart
    state["is_session_restart"] = True
    await agent(state)
    prompt_restart = llm_mock.ainvoke.call_args[0][0]
    print("\n--- Prompt for Restart ---")
    print(prompt_restart)
    assert "returning student after some time away" in prompt_restart
    assert "DO NOT recap previous topics" in prompt_restart

    print("\n✅ Conversational Persona Logic Verified!")

async def test_student_intent_logic():
    print("\nTesting StudentAgent Intent Priority instruction...")
    
    # Mock Retriever and Registry
    llm_mock = AsyncMock()
    retriever_mock = MagicMock()
    agent = StudentAgent(llm_mock, retriever_mock)
    
    # We just want to see the system prompt
    # Signature: query: str, subjects: List[str], target_lang: str, state: Optional[Dict[str, Any]] = None
    prompt = agent._build_student_system_prompt("Friction", ["Physics"], "en", {"student_grade": "A"})
    
    print("\n--- Student Agent System Prompt (Snippet) ---")
    print("\n".join(prompt.split("\n")[10:25]))
    
    assert "EXPLICIT INTENT PRIORITY" in prompt
    assert "NO UNPROMPTED RECAPS" in prompt
    assert "AMBIGUITY HANDLING" in prompt
    
    print("\n✅ Student Intent Rules Verified in Prompt!")

if __name__ == "__main__":
    asyncio.run(test_conversational_persona_logic())
    asyncio.run(test_student_intent_logic())
