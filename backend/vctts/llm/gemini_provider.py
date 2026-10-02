"""Google Gemini via the google-genai SDK."""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import (
    ChatMessage,
    ChatProvider,
    Done,
    GenerationConfig,
    ModelInfo,
    ProviderError,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallsEvent,
    ToolSpec,
)

log = logging.getLogger(__name__)

# Gemini 2.5+/3 models think by default and thinking tokens count against
# max_output_tokens; leave room so short answers aren't cut off.
_THINKING_HEADROOM = 2048
_EFFORT_TO_LEVEL = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "HIGH",
    "max": "HIGH",
}


def to_gemini_contents(messages: list[ChatMessage]) -> list[types.Content]:
    out: list[types.Content] = []

    def append(role: str, parts: list[types.Part]) -> None:
        if not parts:
            return
        if out and out[-1].role == role:
            out[-1].parts = list(out[-1].parts or []) + parts
        else:
            out.append(types.Content(role=role, parts=parts))

    for m in messages:
        if m.role == "user":
            append("user", [types.Part(text=m.content or " ")])
        elif m.role == "assistant":
            if isinstance(m.provider_data, dict) and m.provider_data.get("gemini"):
                # Same-turn replay keeps thought signatures intact.
                append("model", list(m.provider_data["gemini"]))
                continue
            parts: list[types.Part] = []
            if m.content:
                parts.append(types.Part(text=m.content))
            for tc in m.tool_calls:
                parts.append(
                    types.Part(function_call=types.FunctionCall(id=tc.id, name=tc.name, args=tc.arguments))
                )
            append("model", parts)
        elif m.role == "tool":
            key = "error" if m.is_error else "result"
            append(
                "user",
                [types.Part.from_function_response(name=m.name or "tool", response={key: m.content})],
            )
    return out


class GeminiProvider(ChatProvider):
    id = "gemini"
    display_name = "Google Gemini"
    default_model = "gemini-2.5-flash"
    suggested_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError("No API key configured for Gemini")
        self.client = genai.Client(api_key=api_key)

    def _build_config(
        self, config: GenerationConfig, tools: list[ToolSpec] | None, with_thinking: bool
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "max_output_tokens": config.max_tokens + _THINKING_HEADROOM,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if config.system_prompt:
            kwargs["system_instruction"] = config.system_prompt
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if with_thinking and config.reasoning_effort:
            level = _EFFORT_TO_LEVEL.get(config.reasoning_effort.lower())
            if level:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        if tools:
            kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.parameters
                        )
                        for t in tools
                    ]
                )
            ]
        kwargs.update(config.extra)
        return types.GenerateContentConfig(**kwargs)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        contents = to_gemini_contents(messages)
        with_thinking = True
        response = None
        for _attempt in range(2):
            try:
                response = await self.client.aio.models.generate_content_stream(
                    model=config.model,
                    contents=contents,
                    config=self._build_config(config, tools, with_thinking),
                )
                break
            except genai_errors.ClientError as e:
                msg = str(e)
                if with_thinking and config.reasoning_effort and "think" in msg.lower():
                    with_thinking = False  # model doesn't support thinking_level
                    continue
                raise ProviderError(_client_error(e, config.model)) from e
            except genai_errors.APIError as e:
                raise ProviderError(f"Gemini: {e}") from e
        if response is None:
            raise ProviderError("Gemini: request rejected")

        text_parts: list[str] = []
        model_parts: list[types.Part] = []
        tool_calls: list[ToolCall] = []
        stop_reason: str | None = None
        usage: dict[str, int] = {}
        try:
            async for chunk in response:
                if chunk.usage_metadata:
                    usage = {
                        "input_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "output_tokens": chunk.usage_metadata.candidates_token_count or 0,
                    }
                if not chunk.candidates:
                    continue
                cand = chunk.candidates[0]
                if cand.finish_reason:
                    stop_reason = str(cand.finish_reason.value if hasattr(cand.finish_reason, "value") else cand.finish_reason)
                if not cand.content or not cand.content.parts:
                    continue
                for part in cand.content.parts:
                    model_parts.append(part)
                    if part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                id=fc.id or f"call_{uuid.uuid4().hex[:8]}",
                                name=fc.name or "",
                                arguments=dict(fc.args or {}),
                            )
                        )
                    elif part.text and not part.thought:
                        text_parts.append(part.text)
                        yield TextDelta(part.text)
        except genai_errors.APIError as e:
            raise ProviderError(f"Gemini: {e}") from e

        if stop_reason and stop_reason.upper() in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST") and not text_parts:
            raise ProviderError("Gemini blocked this response for safety reasons.")

        if tool_calls:
            yield ToolCallsEvent(
                ChatMessage(
                    role="assistant",
                    content="".join(text_parts),
                    tool_calls=tool_calls,
                    provider_data={"gemini": model_parts},
                )
            )
        yield Done(stop_reason=stop_reason, usage=usage)

    async def list_models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        try:
            pager = await self.client.aio.models.list()
            async for m in pager:
                actions = getattr(m, "supported_actions", None) or []
                if actions and "generateContent" not in actions:
                    continue
                mid = (m.name or "").removeprefix("models/")
                out.append(ModelInfo(id=mid, name=getattr(m, "display_name", None)))
        except genai_errors.ClientError as e:
            raise ProviderError(_client_error(e, "")) from e
        except genai_errors.APIError as e:
            raise ProviderError(f"Gemini: {e}") from e
        return sorted(out, key=lambda m: m.id)


def _client_error(e: genai_errors.ClientError, model: str) -> str:
    code = getattr(e, "code", None)
    if code in (401, 403) or "API key" in str(e):
        return "Gemini: invalid API key"
    if code == 404 and model:
        return f"Gemini: model '{model}' not found"
    return f"Gemini: {getattr(e, 'message', None) or e}"
