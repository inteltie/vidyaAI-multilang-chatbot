import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agents.react_agent import ReActAgent
from tools import ToolRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock tools
@tool
def search_docs(query: str) -> str:
    """Search for documents about a topic."""
    if "photosynthesis" in query.lower():
        return "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
    return "No documents found."

async def verify_agent():
    load_dotenv()
    
    # Setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    registry = ToolRegistry()
    registry.register(search_docs)
    
    agent = ReActAgent(llm, registry, max_iterations=3)
    
    print("\n--- Verifying ReAct Agent (Tool Calling) ---")
    
    # Test 1: Simple query requiring tool
    query = "What is photosynthesis?"
    print(f"\nQuery: {query}")
    result = await agent.run(query, [])
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iterations']}")
    
    if "sunlight" in result['answer'].lower():
        print("✅ Agent successfully used tool and answered.")
    else:
        print("❌ Agent failed to answer correctly.")

    # Test 2: Vague query
    query = "tell me about it"
    print(f"\nQuery: {query}")
    result = await agent.run(query, [])
    print(f"Answer: {result['answer']}")
    
    if "?" in result['answer'] or "what" in result['answer'].lower() or "clarify" in result['answer'].lower():
        print("✅ Agent asked for clarification.")
    else:
        print("❌ Agent did not ask for clarification.")

if __name__ == "__main__":
    asyncio.run(verify_agent())
