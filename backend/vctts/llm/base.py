"""Provider-neutral chat types and the ChatProvider interface.

The message model already carries tool calls and tool results so tools such as
web search can be plugged in later via :mod:`vctts.llm.tools` without touching
the providers or the conversation pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Union

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(id=d["id"], name=d["name"], arguments=d.get("arguments") or {})


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For role == "tool": which call this answers and the tool's name.
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    # Opaque, provider-specific payload used to replay a turn verbatim to the
    # same provider within a single tool loop (e.g. Claude thinking blocks,
    # Gemini thought signatures). Never persisted across turns.
    provider_data: Any = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (type: object)


@dataclass
class GenerationConfig:
    model: str
    system_prompt: str = ""
    max_tokens: int = 1024
    temperature: float | None = None
    # "low" | "medium" | "high" ... mapped per provider; None = provider default.
    reasoning_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextDelta:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallsEvent:
    """Emitted once at the end of a response that requests tool calls."""

    message: ChatMessage
    type: Literal["tool_calls"] = "tool_calls"


@dataclass
class Done:
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    type: Literal["done"] = "done"


StreamEvent = Union[TextDelta, ToolCallsEvent, Done]


@dataclass
class ModelInfo:
    id: str
    name: str | None = None


class ProviderError(RuntimeError):
    """A user-facing error from an LLM provider (bad key, bad model, ...)."""


class ChatProvider(abc.ABC):
    id: str
    display_name: str
    default_model: str
    suggested_models: list[str] = []

    @abc.abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a single model response.

        Yields TextDelta events, then optionally one ToolCallsEvent, then Done.
        """

    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        ...


def drop_unsupported_param(
    error_text: str, params: dict[str, Any], optional: dict[str, tuple[str, ...]]
) -> bool:
    """If an API error complains about one of the optional params, drop it.

    Model capabilities differ a lot (reasoning models reject temperature,
    older models reject effort, ...). Rather than hard-coding a matrix that
    goes stale, we retry without the offending parameter.

    ``optional`` maps a request param name to keywords that identify it in an
    error message. Returns True if something was dropped.
    """
    text = error_text.lower()
    for name, keywords in optional.items():
        if name in params and any(k.lower() in text for k in keywords):
            params.pop(name)
            return True
    return False
