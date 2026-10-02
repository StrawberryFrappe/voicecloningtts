"""Provider tests against mocked HTTP (real SDKs, fake servers)."""

import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from vctts.llm.anthropic_provider import AnthropicProvider, to_anthropic_messages
from vctts.llm.base import ChatMessage, GenerationConfig, ProviderError, TextDelta, ToolCall, ToolCallsEvent, ToolSpec
from vctts.llm.gemini_provider import to_gemini_contents
from vctts.llm.openai_compat import OpenAIProvider, OpenRouterProvider, to_openai_messages

HISTORY = [
    ChatMessage("user", "What is 2+3?"),
    ChatMessage("assistant", "Checking.", [ToolCall("c1", "add", {"a": 2, "b": 3})]),
    ChatMessage("tool", "5", tool_call_id="c1", name="add"),
    ChatMessage("assistant", "It is 5."),
    ChatMessage("user", "Thanks"),
]
TOOLS = [ToolSpec("add", "Add numbers", {"type": "object", "properties": {"a": {"type": "number"}}})]


def sse(events: list[dict], anthropic_style: bool = False) -> bytes:
    out = []
    for e in events:
        if anthropic_style:
            out.append(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n")
        else:
            out.append(f"data: {json.dumps(e)}\n\n")
    if not anthropic_style:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


async def collect(gen):
    return [e async for e in gen]


# -- conversions --------------------------------------------------------------

def test_openai_message_conversion():
    msgs = to_openai_messages(HISTORY, "sys")
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a": 2, "b": 3}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "5"}


def test_anthropic_message_conversion_merges_tool_results():
    msgs = to_anthropic_messages(HISTORY)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"][1] == {"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 2, "b": 3}}
    assert msgs[2]["content"][0]["type"] == "tool_result"


def test_anthropic_replays_raw_blocks_within_turn():
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"},
           {"type": "tool_use", "id": "t1", "name": "add", "input": {}}]
    m = ChatMessage("assistant", "", [ToolCall("t1", "add", {})], provider_data={"anthropic": raw})
    out = to_anthropic_messages([ChatMessage("user", "hi"), m])
    assert out[1]["content"] == raw


def test_gemini_conversion():
    contents = to_gemini_contents(HISTORY)
    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    assert contents[1].parts[1].function_call.name == "add"
    assert contents[2].parts[0].function_response.response == {"result": "5"}


# -- OpenAI / OpenRouter ------------------------------------------------------

def _openai_with(provider, handler):
    provider.client = openai.AsyncOpenAI(
        api_key="sk-test", base_url=str(provider.client.base_url),
        default_headers=dict(provider.client._custom_headers),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


async def test_openai_streams_text_and_tool_calls():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        chunks = [
            {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi "}, "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
             "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                                 "function": {"name": "add", "arguments": "{\"a\":"}}]},
                          "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
             "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 2}"}}]},
                          "finish_reason": "tool_calls"}]},
            {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
             "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
        ]
        return httpx.Response(200, content=sse(chunks), headers={"content-type": "text/event-stream"})

    p = _openai_with(OpenAIProvider("sk-test"), handler)
    cfg = GenerationConfig(model="gpt-test", system_prompt="be brief", max_tokens=50, temperature=0.3,
                           reasoning_effort="low")
    events = await collect(p.stream(HISTORY, cfg, TOOLS))
    assert isinstance(events[0], TextDelta) and events[0].text == "Hi "
    tc = next(e for e in events if isinstance(e, ToolCallsEvent))
    assert tc.message.tool_calls[0].arguments == {"a": 2}
    assert events[-1].usage == {"input_tokens": 7, "output_tokens": 3}
    body = seen["body"]
    assert body["max_completion_tokens"] == 50 and body["reasoning_effort"] == "low"
    assert body["tools"][0]["function"]["name"] == "add"
    assert body["stream"] is True


async def test_openai_drops_unsupported_temperature_and_retries():
    bodies = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model.",
                                                       "type": "invalid_request_error", "param": "temperature"}})
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(200, content=sse([chunk]), headers={"content-type": "text/event-stream"})

    p = _openai_with(OpenAIProvider("sk"), handler)
    events = await collect(p.stream([ChatMessage("user", "hi")], GenerationConfig(model="o", temperature=0.2)))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["ok"]
    assert len(bodies) == 2 and "temperature" not in bodies[1]


async def test_openai_auth_error_is_friendly():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "bad key", "type": "invalid_request_error"}})

    p = _openai_with(OpenAIProvider("sk"), handler)
    with pytest.raises(ProviderError, match="invalid API key"):
        await collect(p.stream([ChatMessage("user", "hi")], GenerationConfig(model="m")))


async def test_openrouter_uses_max_tokens_and_reasoning_body():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "yo"}, "finish_reason": "stop"}]}
        return httpx.Response(200, content=sse([chunk]), headers={"content-type": "text/event-stream"})

    p = _openai_with(OpenRouterProvider("or-key"), handler)
    await collect(p.stream([ChatMessage("user", "hi")], GenerationConfig(model="openrouter/auto", max_tokens=99,
                                                                          reasoning_effort="low")))
    assert seen["body"]["max_tokens"] == 99
    assert seen["body"]["reasoning"] == {"effort": "low"}
    assert seen["headers"]["x-title"] == "VoiceCloningTTS"


# -- Anthropic ------------------------------------------------------------------

def _anthropic_events(text_parts, tool=None, stop="end_turn"):
    ev = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                                "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                                                "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 1}}}]
    idx = 0
    if text_parts:
        ev.append({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
        for t in text_parts:
            ev.append({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": t}})
        ev.append({"type": "content_block_stop", "index": idx})
        idx += 1
    if tool:
        ev.append({"type": "content_block_start", "index": idx,
                   "content_block": {"type": "tool_use", "id": "toolu_1", "name": tool, "input": {}}})
        ev.append({"type": "content_block_delta", "index": idx,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"a\": 2, \"b\": 3}"}})
        ev.append({"type": "content_block_stop", "index": idx})
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
               "usage": {"output_tokens": 9}})
    ev.append({"type": "message_stop"})
    return ev


def _anthropic_with(handler):
    p = AnthropicProvider("sk-ant-test")
    # anthropic>=1.0 is built on httpx2
    p.client = anthropic.AsyncAnthropic(api_key="sk-ant-test",
                                        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return p


async def test_anthropic_streams_text_with_effort_and_fallbacks():
    seen = {}

    def handler(request: httpx2.Request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx2.Response(200, content=sse(_anthropic_events(["Hello", " world."]), True),
                              headers={"content-type": "text/event-stream"})

    p = _anthropic_with(handler)
    cfg = GenerationConfig(model="claude-opus-5-5", system_prompt="sys", max_tokens=300, reasoning_effort="low")
    events = await collect(p.stream([ChatMessage("user", "hi")], cfg))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello world."
    assert events[-1].stop_reason == "end_turn"
    body = seen["body"]
    assert body["system"] == "sys" and body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == 300 + 4096  # thinking headroom on always-thinking models
    assert body["fallbacks"] == "default" and "server-side-fallback" in seen["beta"]
    assert "temperature" not in body


async def test_anthropic_tool_use_and_param_retry():
    bodies = []

    def handler(request: httpx2.Request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx2.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "temperature is not supported for this model"}})
        return httpx2.Response(200, content=sse(_anthropic_events(["Adding."], tool="add", stop="tool_use"), True),
                              headers={"content-type": "text/event-stream"})

    p = _anthropic_with(handler)
    cfg = GenerationConfig(model="claude-haiku-4-5", max_tokens=100, temperature=0.5)
    events = await collect(p.stream([ChatMessage("user", "add")], cfg, TOOLS))
    assert len(bodies) == 2 and "temperature" not in bodies[1]
    assert bodies[1]["tools"][0]["input_schema"]["type"] == "object"
    assert bodies[1]["max_tokens"] == 100 and "fallbacks" not in bodies[1]
    tc = next(e for e in events if isinstance(e, ToolCallsEvent))
    assert tc.message.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert tc.message.provider_data["anthropic"][1]["type"] == "tool_use"


async def test_anthropic_refusal_raises():
    def handler(request):
        return httpx2.Response(200, content=sse(_anthropic_events([], stop="refusal"), True),
                              headers={"content-type": "text/event-stream"})

    with pytest.raises(ProviderError, match="declined"):
        await collect(_anthropic_with(handler).stream([ChatMessage("user", "x")], GenerationConfig(model="claude-haiku-4-5")))
