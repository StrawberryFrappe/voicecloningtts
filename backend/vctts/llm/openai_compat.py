"""OpenAI and OpenRouter (OpenAI-compatible Chat Completions API)."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import openai
from openai import AsyncOpenAI

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


def to_openai_messages(messages: list[ChatMessage], system_prompt: str) -> list[dict]:
    out: list[dict] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            elif not m.content:
                msg["content"] = ""
            out.append(msg)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OpenAICompatProvider(ChatProvider):
    """Chat Completions streaming; shared by OpenAI and OpenRouter."""

    max_tokens_param = "max_completion_tokens"

    def __init__(self, api_key: str, base_url: str | None = None, default_headers: dict | None = None):
        if not api_key:
            raise ProviderError(f"No API key configured for {self.display_name}")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    def _effort_params(self, effort: str) -> dict[str, Any]:
        return {"reasoning_effort": effort}

    def _build_params(
        self, messages: list[ChatMessage], config: GenerationConfig, tools: list[ToolSpec] | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": config.model,
            "messages": to_openai_messages(messages, config.system_prompt),
            self.max_tokens_param: config.max_tokens,
        }
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.reasoning_effort:
            params.update(self._effort_params(config.reasoning_effort))
        if tools:
            params["tools"] = to_openai_tools(tools)
        params.update(config.extra)
        return params

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        params = self._build_params(messages, config, tools)
        response = None
        for _attempt in range(4):
            try:
                response = await self.client.chat.completions.create(
                    **params, stream=True, stream_options={"include_usage": True}
                )
                break
            except openai.BadRequestError as e:
                if drop_unsupported_param(
                    str(e),
                    params,
                    {
                        "temperature": ("temperature",),
                        "reasoning_effort": ("reasoning_effort", "reasoning effort"),
                        "extra_body": ("reasoning",),
                    },
                ):
                    log.info("retrying %s without an unsupported parameter: %s", config.model, e)
                    continue
                raise ProviderError(f"{self.display_name}: {e.message}") from e
            except openai.AuthenticationError as e:
                raise ProviderError(f"{self.display_name}: invalid API key") from e
            except openai.NotFoundError as e:
                raise ProviderError(f"{self.display_name}: model '{config.model}' not found") from e
            except openai.APIError as e:
                raise ProviderError(f"{self.display_name}: {e.message}") from e
        if response is None:
            raise ProviderError(f"{self.display_name}: request rejected")

        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        stop_reason: str | None = None
        usage: dict[str, int] = {}
        try:
            async for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None:
                    if delta.content:
                        text_parts.append(delta.content)
                        yield TextDelta(delta.content)
                    for tc in delta.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    stop_reason = choice.finish_reason
        except openai.APIError as e:
            raise ProviderError(f"{self.display_name}: {e.message}") from e

        if calls:
            tool_calls = []
            for i in sorted(calls):
                c = calls[i]
                try:
                    args = json.loads(c["arguments"]) if c["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": c["arguments"]}
                tool_calls.append(ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=args))
            yield ToolCallsEvent(
                ChatMessage(role="assistant", content="".join(text_parts), tool_calls=tool_calls)
            )
        yield Done(stop_reason=stop_reason, usage=usage)

    async def list_models(self) -> list[ModelInfo]:
        try:
            page = await self.client.models.list()
        except openai.AuthenticationError as e:
            raise ProviderError(f"{self.display_name}: invalid API key") from e
        except openai.APIError as e:
            raise ProviderError(f"{self.display_name}: {e.message}") from e
        out = [ModelInfo(id=m.id, name=getattr(m, "name", None)) async for m in page]
        return sorted(out, key=lambda m: m.id)


class OpenAIProvider(OpenAICompatProvider):
    id = "openai"
    display_name = "OpenAI"
    default_model = "gpt-5-mini"
    suggested_models = ["gpt-5-mini", "gpt-5", "gpt-4.1-mini"]

    def __init__(self, api_key: str):
        super().__init__(api_key)

    async def list_models(self) -> list[ModelInfo]:
        models = await super().list_models()
        # The raw list includes embeddings, TTS, image models, etc.
        skip = ("embedding", "tts", "whisper", "dall-e", "image", "moderation", "transcribe", "audio", "realtime", "search")
        return [m for m in models if not any(s in m.id for s in skip)]


class OpenRouterProvider(OpenAICompatProvider):
    id = "openrouter"
    display_name = "OpenRouter"
    default_model = "openrouter/auto"
    suggested_models = ["openrouter/auto"]
    max_tokens_param = "max_tokens"

    def __init__(self, api_key: str):
        super().__init__(
            api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/StrawberryFrappe/voicecloningtts",
                "X-Title": "VoiceCloningTTS",
            },
        )

    def _effort_params(self, effort: str) -> dict[str, Any]:
        # OpenRouter's unified reasoning parameter.
        return {"extra_body": {"reasoning": {"effort": effort}}}
