"""Tool registry: the extension point for web search and friends.

Nothing is registered by default. To add a tool later::

    registry.register(
        ToolSpec("web_search", "Search the web", {"type": "object", "properties": {...}}),
        handler=my_async_search,
    )

The conversation pipeline passes ``registry.specs()`` to the provider and
executes any calls the model makes via :meth:`ToolRegistry.execute`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from .base import ChatMessage, ToolCall, ToolSpec

log = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any] | Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}
        self._enabled: set[str] = set()

    def register(self, spec: ToolSpec, handler: ToolHandler, enabled: bool = True) -> None:
        self._tools[spec.name] = (spec, handler)
        if enabled:
            self._enabled.add(spec.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._enabled.discard(name)

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._tools:
            raise KeyError(name)
        (self._enabled.add if enabled else self._enabled.discard)(name)

    def list(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "enabled": s.name in self._enabled}
            for s, _ in self._tools.values()
        ]

    def specs(self, allowed: list[str] | None = None) -> list[ToolSpec]:
        names = self._enabled if allowed is None else self._enabled & set(allowed)
        return [self._tools[n][0] for n in sorted(names)]

    async def execute(self, call: ToolCall) -> ChatMessage:
        entry = self._tools.get(call.name)
        if entry is None:
            return ChatMessage(
                role="tool", tool_call_id=call.id, name=call.name,
                content=f"Unknown tool: {call.name}", is_error=True,
            )
        _, handler = entry
        try:
            result = handler(call.arguments)
            if asyncio.iscoroutine(result):
                result = await result
            content = result if isinstance(result, str) else json.dumps(result, default=str)
            return ChatMessage(role="tool", tool_call_id=call.id, name=call.name, content=content)
        except Exception as e:  # tool errors go back to the model, not the user
            log.exception("tool %s failed", call.name)
            return ChatMessage(
                role="tool", tool_call_id=call.id, name=call.name,
                content=f"Tool error: {e}", is_error=True,
            )
