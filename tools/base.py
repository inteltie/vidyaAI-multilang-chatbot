"""Base tool interface for ReAct agent."""

from typing import Any, Dict, Protocol
from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all tools that the ReAct agent can use."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema describing the tool's parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        Execute the tool with given parameters.
        
        Returns:
            String observation that will be shown to the agent
        """
        pass
    
    def format_for_prompt(self) -> str:
        """Format tool information for inclusion in reasoning prompt."""
        params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in self.parameters_schema.items())
        return f"{self.name}({params}): {self.description}"


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]
    
    def list_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def format_for_prompt(self) -> str:
        """Format all tools for inclusion in reasoning prompt."""
        return "\n".join(f"- {tool.format_for_prompt()}" for tool in self._tools.values())


__all__ = ["Tool", "ToolRegistry"]
