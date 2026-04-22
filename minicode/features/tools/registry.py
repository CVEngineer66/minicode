from __future__ import annotations

from typing import Iterable

from .types import ToolSpec


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def add(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def tool_schemas(self) -> list[dict]:
        return [tool.to_model_schema() for tool in self.list()]
