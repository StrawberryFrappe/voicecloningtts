"""Claude via the official Anthropic SDK (Messages API, streaming)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import anthropic

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
    drop_unsupported_param,
)

log = logging.getLogger(__name__)

# Models that think by default (adaptive thinking can't be turned off). Their
# thinking tokens count against max_tokens, so we add headroom on top of the
# persona's answer-length cap; the answer length itself is steered by the
# system prompt and the effort level.
_ALWAYS_THINKING_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-fable", "claude-mythos")
_THINKING_HEADROOM = 4096

# Models that accept the server-side refusal fallback ("fallbacks": "default").
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic_messages(messages: list[ChatMessage]) -> list[dict]:
    out: list[dict] = []

    def append(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m.role == "user":
            append("user", [{"type": "text", "text": m.content or " "}])
        elif m.role == "assistant":
            if m.provider_data and isinstance(m.provider_data, dict) and m.provider_data.get("anthropic"):
                # Same-turn replay inside a tool loop: send the blocks back unchanged
                # (keeps thinking blocks and their signatures intact).
                append("assistant", list(m.provider_data["anthropic"]))
                continue
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            append("assistant", blocks)
        elif m.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            if m.is_error:
                block["is_error"] = True
            append("user", [block])
    return out


class AnthropicProvider(ChatProvider):
    id = "anthropic"
    display_name = "Claude (Anthropic)"
    default_model = "claude-opus-5-5"
    suggested_models = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"]

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError("No API key configured for Claude")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _build_params(
        self, messages: list[ChatMessage], config: GenerationConfig, tools: list[ToolSpec] | None
    ) -> dict[str, Any]:
        max_tokens = config.max_tokens
        if config.model.startswith(_ALWAYS_THINKING_PREFIXES):
            max_tokens += _THINKING_HEADROOM
        params: dict[str, Any] = {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": to_anthropic_messages(messages),
        }
        if config.system_prompt:
            params["system"] = config.system_prompt
        extra_body: dict[str, Any] = {}
        if config.temperature is not None:
            # anthropic>=1.0 dropped the typed `temperature` argument (current
            # Claude models reject sampling params); older models still take it.
            extra_body["temperature"] = config.temperature
        if config.reasoning_effort:
            params["output_config"] = {"effort": config.reasoning_effort}
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if config.model.startswith(_FALLBACK_PREFIXES):
            # If a safety classifier declines, let the API retry on a fallback model.
            params["extra_headers"] = {"anthropic-beta": _FALLBACK_BETA}
            extra_body["fallbacks"] = "default"
        if extra_body:
            params["extra_body"] = extra_body
        params.update(config.extra)
        return params

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        params = self._build_params(messages, config, tools)
        for _attempt in range(4):
            emitted = False
            try:
                async with self.client.messages.stream(**params) as stream:
                    async for event in stream:
                        if event.type == "text" and event.text:
                            emitted = True
                            yield TextDelta(event.text)
                    final = await stream.get_final_message()
            except anthropic.BadRequestError as e:
                if not emitted and _drop_rejected(str(e), params):
                    log.info("retrying %s without an unsupported parameter: %s", config.model, e)
                    continue
                raise ProviderError(f"Claude: {_err_message(e)}") from e
            except anthropic.AuthenticationError as e:
                raise ProviderError("Claude: invalid API key") from e
            except anthropic.NotFoundError as e:
                raise ProviderError(f"Claude: model '{config.model}' not found") from e
            except anthropic.RateLimitError as e:
                raise ProviderError("Claude: rate limited, try again in a moment") from e
            except anthropic.APIStatusError as e:
                raise ProviderError(f"Claude: {_err_message(e)}") from e
            except anthropic.APIConnectionError as e:
                raise ProviderError("Claude: connection error") from e
            break
        else:
            raise ProviderError("Claude: request rejected")

        if final.stop_reason == "refusal":
            raise ProviderError("Claude declined to answer this request.")

        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in final.content
            if b.type == "tool_use"
        ]
        if tool_calls:
            text = "".join(b.text for b in final.content if b.type == "text")
            raw = [b.model_dump(exclude_none=True) for b in final.content]
            yield ToolCallsEvent(
                ChatMessage(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls,
                    provider_data={"anthropic": raw},
                )
            )
        usage = {
            "input_tokens": getattr(final.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(final.usage, "output_tokens", 0) or 0,
        }
        yield Done(stop_reason=final.stop_reason, usage=usage)

    async def list_models(self) -> list[ModelInfo]:
        try:
            page = await self.client.models.list()
            return [ModelInfo(id=m.id, name=getattr(m, "display_name", None)) async for m in page]
        except anthropic.AuthenticationError as e:
            raise ProviderError("Claude: invalid API key") from e
        except anthropic.APIError as e:
            raise ProviderError(f"Claude: {_err_message(e)}") from e


def _drop_rejected(error_text: str, params: dict[str, Any]) -> bool:
    """Remove the optional parameter an API error complains about (see base.drop_unsupported_param)."""
    extra = params.get("extra_body") or {}
    if drop_unsupported_param(error_text, extra, {"temperature": ("temperature",),
                                                  "fallbacks": ("fallback",)}):
        if "fallbacks" not in extra:
            params.pop("extra_headers", None)
        if not extra:
            params.pop("extra_body", None)
        return True
    return drop_unsupported_param(error_text, params, {"output_config": ("effort", "output_config")})


def _err_message(e: Exception) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return getattr(e, "message", None) or str(e)
