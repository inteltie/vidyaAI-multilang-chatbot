"""Base protocol for all agents."""

from typing import Protocol
from state import AgentState


class Agent(Protocol):
    """Base protocol that all agents must implement."""
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process the agent state and return updated state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with response
        """
        ...
