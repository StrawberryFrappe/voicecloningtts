import asyncio

import pytest

from vctts.audio.sinks import MemorySink
from vctts.conversation import ConversationService, trim_history
from vctts.events import EventBus
from vctts.llm import ChatMessage, ProviderRegistry, TextDelta, ToolRegistry, ToolSpec
from vctts.llm.base import Done, ProviderError
from vctts.personas import Persona, PersonaStore
from vctts.secrets import SecretStore
from vctts.storage import Database
from vctts.tts import TTSManager
from vctts.voices import VoiceLibrary

from .conftest import FakeEngine, FakeProvider, tool_call_script


class Env:
    def __init__(self, paths, provider, speech_wav):
        self.db = Database(paths.db)
        self.personas = PersonaStore(self.db)
        self.providers = ProviderRegistry(SecretStore(paths.root / "s.json", use_keyring=False))
        self.providers.register_override("fake", provider)
        self.tools = ToolRegistry()
        self.voices = VoiceLibrary(paths.voices)
        self.tts = TTSManager(self.voices)
        self.engine = FakeEngine()
        self.tts.register_engine(self.engine)
        self.sink = MemorySink()
        self.bus = EventBus()
        self.events = self.bus.subscribe()
        self.svc = ConversationService(self.db, self.personas, self.providers, self.tools, self.tts,
                                       self.voices, self.sink, self.bus)
        v, _ = self.voices.create_from_audio(speech_wav, "Me")
        self.db.set_setting("active_voice_id", v.id)
        p = Persona(id="p1", name="Test", provider="fake", model="fake-1", length="short")
        self.personas.save(p)
        self.personas.set_active("p1")
        self.conv = self.db.create_conversation(persona_id="p1")

    def drain(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out


@pytest.fixture
def env(paths, speech_wav):
    e = Env(paths, FakeProvider(), speech_wav)
    yield e
    e.tts.shutdown()


async def test_turn_streams_text_and_speaks_sentences_in_order(env):
    turn = await env.svc.send(env.conv["id"], "hi")
    await turn.task
    msgs = env.db.list_messages(env.conv["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "Hello there. This is a test reply! Bye."
    assert msgs[1]["meta"]["model"] == "fake-1"
    # speech starts before the reply is complete; very short openers merge with the next sentence
    assert env.engine.spoken[0] == "Hello there. This is a test reply!"
    assert " ".join(env.engine.spoken) == "Hello there. This is a test reply! Bye."
    assert [s[0] for s in env.sink.segments] == [f"{turn.id}:{i}" for i in range(len(env.engine.spoken))]
    types = [e["type"] for e in env.drain()]
    assert types.index("assistant_start") < types.index("delta") < types.index("assistant_done")
    assert "tts_chunk" in types and types[-1] == "tts_done"
    assert env.db.get_conversation(env.conv["id"])["title"] == "hi"


async def test_system_prompt_and_history_passed_to_provider(env):
    await (await env.svc.send(env.conv["id"], "first")).task
    await (await env.svc.send(env.conv["id"], "second")).task
    calls = env.providers.get("fake").calls
    msgs, cfg, tools = calls[-1]
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert "brief" in cfg.system_prompt and cfg.max_tokens == 300
    assert tools is None  # no tools registered yet


async def test_speak_false_skips_tts(env):
    await (await env.svc.send(env.conv["id"], "hi", speak=False)).task
    assert env.engine.spoken == [] and env.sink.segments == []


async def test_tool_loop(paths, speech_wav):
    e = Env(paths, FakeProvider(tool_call_script()), speech_wav)
    e.tools.register(ToolSpec("add", "add", {"type": "object"}), lambda args: args["a"] + args["b"])
    await (await e.svc.send(e.conv["id"], "2+3?")).task
    msgs = e.db.list_messages(e.conv["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"][0]["name"] == "add" and msgs[2]["content"] == "5"
    assert msgs[3]["content"] == "The answer is five."
    second_call_msgs = e.providers.get("fake").calls[1][0]
    assert [m.role for m in second_call_msgs][-2:] == ["assistant", "tool"]
    assert "Let me add." in e.engine.spoken[0]
    e.tts.shutdown()


async def test_stop_interrupts_and_keeps_partial(paths, speech_wav):
    slow = FakeProvider([[TextDelta("One two three. "), TextDelta("Four five six. "), TextDelta("Seven."),
                          Done("stop")]], delay=0.2)
    e = Env(paths, slow, speech_wav)
    turn = await e.svc.send(e.conv["id"], "count")
    await asyncio.sleep(0.3)
    await e.svc.stop()
    assert turn.task.done()
    msgs = e.db.list_messages(e.conv["id"])
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["meta"].get("interrupted")
    assert "Seven" not in msgs[-1]["content"]
    assert e.sink.stopped >= 1
    e.tts.shutdown()


async def test_provider_error_is_reported(paths, speech_wav):
    class Broken(FakeProvider):
        async def stream(self, messages, config, tools=None):
            raise ProviderError("Claude: invalid API key")
            yield  # pragma: no cover

    e = Env(paths, Broken(), speech_wav)
    await (await e.svc.send(e.conv["id"], "hi")).task
    errors = [ev for ev in e.drain() if ev["type"] == "error"]
    assert errors and "invalid API key" in errors[0]["message"]
    e.tts.shutdown()


async def test_regenerate_replaces_last_reply(env):
    await (await env.svc.send(env.conv["id"], "hi")).task
    await (await env.svc.regenerate(env.conv["id"])).task
    roles = [m["role"] for m in env.db.list_messages(env.conv["id"])]
    assert roles == ["user", "assistant"]


async def test_speak_text_without_llm(env):
    text = "Hola a todos, bienvenidos al canal. Esto es una prueba de la voz clonada con varias frases."
    turn = await env.svc.speak_text(text)
    await turn.task
    assert env.engine.spoken == ["Hola a todos, bienvenidos al canal.",
                                 "Esto es una prueba de la voz clonada con varias frases."]
    assert len(env.sink.segments) == 2


def test_trim_history_starts_with_user():
    msgs = [ChatMessage("user", "a"), ChatMessage("assistant", "b"), ChatMessage("tool", "c"),
            ChatMessage("assistant", "d"), ChatMessage("user", "e")]
    assert [m.content for m in trim_history(msgs, 3)] == ["e"]
    assert [m.content for m in trim_history(msgs, 0)] == ["a", "b", "c", "d", "e"]
