"""
A pydantic-based tool abstraction that lets agents discover and invoke tools by name and description.
Each tool describes its name, description, and parameter schema and a callable function.
The registry supports registration, discovery (for prompt construction), and execution with error handling.
"""

from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional

_TYPE_MAP = {"str": str, "int": int, "float": float, "bool": bool}


class ToolParameter(BaseModel):
    name: str
    type: str  # type name as string: "str", "int", "float", "bool"
    description: str
    required: bool = True

class Tool(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    func: Callable

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def execute_tool(self, name: str, **kwargs) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool with name '{name}' not found.")
        
        # Validate parameters
        for param in tool.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter '{param.name}' for tool '{name}'.")
            expected = _TYPE_MAP.get(param.type)
            if expected and param.name in kwargs and not isinstance(kwargs[param.name], expected):
                raise ValueError(f"Parameter '{param.name}' must be of type {param.type} for tool '{name}'.")

        try:
            return tool.func(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{name}': {str(e)}") from e
        
    def generate_tools_prompt(self) -> str:
        prompt = "Available tools:\n"
        for tool in self.list_tools():
            prompt += f"- {tool.name}: {tool.description}\n"
            for param in tool.parameters:
                prompt += f"  - {param.name} ({param.type}): {param.description} {'(required)' if param.required else '(optional)'}\n"
        return prompt