"""Node wrapper for image generation agent."""

from time import perf_counter

from agents.image_generation_agent import ImageGenerationAgent
from state import AgentState


class ImageGenerationAgentNode:
    """LangGraph node: image_generation_agent."""

    def __init__(self, agent: ImageGenerationAgent) -> None:
        self._agent = agent

    async def __call__(self, state: AgentState) -> dict:
        start = perf_counter()
        updates = await self._agent(state)
        updates["timings"] = {"image_generation_agent": perf_counter() - start}
        return updates


__all__ = ["ImageGenerationAgentNode"]
