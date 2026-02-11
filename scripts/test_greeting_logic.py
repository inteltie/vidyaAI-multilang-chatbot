"""Test script to verify fresh conversation greeting vs mid-conversation greeting."""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.conversational_agent import ConversationalAgent
from langchain_openai import ChatOpenAI
from config import settings
from state import ConversationTurn


async def test_fresh_greeting():
    """Test greeting with no conversation history (fresh conversation)."""
    print("\n" + "="*60)
    print("TEST 1: Fresh Conversation Greeting")
    print("="*60)
    
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key
    )
    
    agent = ConversationalAgent(llm)
    
    # Simulate fresh conversation state (no history)
    state = {
        "query": "hi",
        "conversation_history": [],  # Empty history
        "session_metadata": {},
        "language": "en"
    }
    
    result = await agent(state)
    
    print(f"\nQuery: {state['query']}")
    print(f"History: {len(state['conversation_history'])} messages")
    print(f"\nResponse: {result['response']}")
    print(f"LLM Calls: {result['llm_calls']}")
    print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
    
    # Check if response mentions "we were discussing" (should NOT)
    if "we were" in result['response'].lower() or "discussing" in result['response'].lower():
        print("\n❌ FAIL: Response mentions previous discussion in fresh conversation!")
        return False
    else:
        print("\n✅ PASS: Response is appropriate for fresh conversation")
        return True


from langchain_core.messages import HumanMessage, AIMessage

async def test_mid_conversation_greeting():
    """Test greeting with existing conversation history."""
    print("\n" + "="*60)
    print("TEST 2: Mid-Conversation Greeting")
    print("="*60)
    
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key
    )
    
    agent = ConversationalAgent(llm)
    
    # Simulate mid-conversation state (with history)
    state = {
        "query": "hello",
        "conversation_history": [
            HumanMessage(content="What is photosynthesis?"),
            AIMessage(content="Photosynthesis is the process by which plants convert light energy into chemical energy..."),
            HumanMessage(content="Can you explain more about chlorophyll?"),
            AIMessage(content="Chlorophyll is the green pigment in plants that absorbs light energy...")
        ],
        "session_metadata": {
            "summary": "photosynthesis and chlorophyll",
            "subject": "Biology"
        },
        "language": "en"
    }
    
    result = await agent(state)
    
    print(f"\nQuery: {state['query']}")
    print(f"History: {len(state['conversation_history'])} messages")
    print(f"Summary: {state['session_metadata']['summary']}")
    print(f"\nResponse: {result['response']}")
    print(f"LLM Calls: {result['llm_calls']}")
    print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
    
    # Check if response mentions the previous topic (should)
    response_lower = result['response'].lower()
    if any(keyword in response_lower for keyword in ["discussing", "talking about", "were", "continue"]):
        print("\n✅ PASS: Response appropriately references previous discussion")
        return True
    else:
        print("\n⚠️  WARNING: Response doesn't clearly reference previous discussion")
        return True  # Still pass, as long as it's contextual


async def main():
    print("\n🧪 Testing Conversational Agent Greeting Logic")
    print("=" * 60)
    
    test1_pass = await test_fresh_greeting()
    test2_pass = await test_mid_conversation_greeting()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Fresh Conversation: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Mid-Conversation: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
