"""UI-agnostic conversation pipeline.

    user text ─► LLM stream ─► sentence chunker ─► TTS (worker thread) ─► SinkRouter
                    │                                                     ├─ virtual mic mixer
                    └─► tool calls ─► ToolRegistry ─► back to the LLM      ├─ browser preview
                                                                          └─ (future) Discord

Any frontend (the desktop UI over WebSocket, a future Discord bot) drives the
same :class:`ConversationService` and listens to the same :class:`EventBus`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .audio.sinks import AudioSink
from .events import EventBus
from .llm import (
    ChatMessage,
    ProviderError,
    ProviderRegistry,
    TextDelta,
    ToolCall,
    ToolCallsEvent,
    ToolRegistry,
)
from .personas import PersonaOverrides, PersonaStore, build_generation
from .storage import Database, new_id
from .tts import CancelToken, SentenceChunker, TTSError, TTSManager, split_text
from .voices import Voice, VoiceLibrary

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6


@dataclass
class Turn:
    id: str
    conversation_id: str | None
    cancel: CancelToken = field(default_factory=CancelToken)
    task: asyncio.Task | None = None
    started: float = field(default_factory=time.time)


def db_to_chat(rows: list[dict]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for r in rows:
        role = r["role"]
        if role not in ("user", "assistant", "tool"):
            continue
        if role == "assistant" and not r["content"] and not r["tool_calls"]:
            continue
        out.append(
            ChatMessage(
                role=role,
                content=r["content"],
                tool_calls=[ToolCall.from_dict(t) for t in r["tool_calls"]],
                tool_call_id=r["tool_call_id"],
                name=r["name"],
                is_error=bool(r["meta"].get("is_error")),
            )
        )
    return out


def trim_history(messages: list[ChatMessage], limit: int) -> list[ChatMessage]:
    """Keep the last ``limit`` messages without splitting a tool-call group and
    always starting on a user message."""
    if limit <= 0 or len(messages) <= limit:
        msgs = messages
    else:
        msgs = messages[-limit:]
    while msgs and msgs[0].role != "user":
        msgs = msgs[1:]
    return msgs


class ConversationService:
    def __init__(
        self,
        db: Database,
        personas: PersonaStore,
        providers: ProviderRegistry,
        tools: ToolRegistry,
        tts: TTSManager,
        voices: VoiceLibrary,
        sink: AudioSink,
        bus: EventBus,
    ):
        self.db = db
        self.personas = personas
        self.providers = providers
        self.tools = tools
        self.tts = tts
        self.voices = voices
        self.sink = sink
        self.bus = bus
        self._turns: dict[str, Turn] = {}
        self._speech_lock = asyncio.Lock()

    # -- settings helpers ------------------------------------------------
    @property
    def history_limit(self) -> int:
        return int(self.db.get_setting("history_limit", 40))

    def active_voice(self, persona_voice_id: str | None = None) -> Voice | None:
        return self.voices.get(persona_voice_id) or self.voices.get(self.db.get_setting("active_voice_id"))

    async def _emit(self, event: dict[str, Any]) -> None:
        await self.bus.publish(event)

    # -- public API ------------------------------------------------------
    def busy(self) -> bool:
        return any(t.task and not t.task.done() for t in self._turns.values())

    async def send(
        self,
        conversation_id: str,
        text: str,
        persona_id: str | None = None,
        overrides: PersonaOverrides | None = None,
        speak: bool | None = None,
        interrupt: bool = True,
    ) -> Turn:
        conv = self.db.get_conversation(conversation_id)
        if conv is None:
            raise KeyError(conversation_id)
        if interrupt:
            await self.stop()
        if persona_id and persona_id != conv["persona_id"]:
            conv = self.db.update_conversation(conversation_id, persona_id=persona_id)
            await self._emit({"type": "conversation_updated", "conversation": conv})
        user_msg = self.db.add_message(conversation_id, "user", text)
        await self._emit({"type": "message", "conversation_id": conversation_id, "message": user_msg})
        if conv["title"] in ("New chat", "") and text.strip():
            title = text.strip().splitlines()[0][:48]
            conv = self.db.update_conversation(conversation_id, title=title)
            await self._emit({"type": "conversation_updated", "conversation": conv})
        return self._start_turn(conversation_id, conv.get("persona_id"), overrides, speak)

    async def regenerate(
        self, conversation_id: str, overrides: PersonaOverrides | None = None, speak: bool | None = None
    ) -> Turn:
        """Drop the last assistant reply (and its tool calls) and answer again."""
        await self.stop()
        rows = self.db.list_messages(conversation_id)
        last_user = max((r["seq"] for r in rows if r["role"] == "user"), default=None)
        if last_user is None:
            raise ValueError("Nothing to regenerate")
        self.db.delete_messages_from(conversation_id, last_user + 1)
        conv = self.db.get_conversation(conversation_id)
        await self._emit({"type": "history_changed", "conversation_id": conversation_id})
        return self._start_turn(conversation_id, conv and conv.get("persona_id"), overrides, speak)

    def _start_turn(self, conversation_id, persona_id, overrides, speak) -> Turn:
        turn = Turn(id=new_id(), conversation_id=conversation_id)
        turn.task = asyncio.create_task(self._run_turn(turn, persona_id, overrides, speak))
        self._turns[turn.id] = turn
        turn.task.add_done_callback(lambda _t, tid=turn.id: self._turns.pop(tid, None))
        return turn

    async def stop(self) -> None:
        """Stop generation and speech everywhere (the big red button)."""
        turns = list(self._turns.values())
        for t in turns:
            t.cancel.cancel()
            if t.task and not t.task.done():
                t.task.cancel()
        for t in turns:
            if t.task:
                try:
                    await asyncio.wait_for(asyncio.shield(t.task), timeout=5)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
        self.sink.stop()
        await self._emit({"type": "stopped"})

    async def speak_text(
        self, text: str, voice_id: str | None = None, language: str | None = None, interrupt: bool = False
    ) -> Turn:
        """Speak arbitrary text in the active voice (no LLM)."""
        if interrupt:
            await self.stop()
        voice = self.voices.get(voice_id) or self.active_voice()
        if voice is None:
            raise ValueError("Create or select a voice first.")
        turn = Turn(id=new_id(), conversation_id=None)

        async def run() -> None:
            queue: asyncio.Queue = asyncio.Queue()
            for chunk in split_text(text):
                queue.put_nowait(chunk)
            queue.put_nowait(None)
            await self._speak(turn, queue, voice, language)

        turn.task = asyncio.create_task(run())
        self._turns[turn.id] = turn
        turn.task.add_done_callback(lambda _t, tid=turn.id: self._turns.pop(tid, None))
        return turn

    # -- internals -------------------------------------------------------
    async def _run_turn(self, turn: Turn, persona_id, overrides, speak) -> None:
        cid = turn.conversation_id
        assert cid is not None
        persona = self.personas.resolve(persona_id)
        speak = persona.speak_replies if speak is None else speak
        speech_queue: asyncio.Queue | None = None
        speech_task: asyncio.Task | None = None
        voice = self.active_voice(persona.voice_id) if speak else None
        if speak and voice is None:
            await self._emit({"type": "warning", "conversation_id": cid,
                              "message": "No voice selected; replying in text only."})
        if voice is not None:
            speech_queue = asyncio.Queue()
            speech_task = asyncio.create_task(
                self._speak(turn, speech_queue, voice, persona.tts_language)
            )

        chunker = SentenceChunker()
        reply_text = ""
        await self._emit({"type": "assistant_start", "conversation_id": cid, "turn_id": turn.id,
                          "persona_id": persona.id})
        try:
            provider_id, gen = build_generation(persona, overrides)
            provider = self.providers.get(provider_id)
            history = trim_history(db_to_chat(self.db.list_messages(cid)), self.history_limit)
            tool_specs = self.tools.specs() or None

            for _round in range(MAX_TOOL_ROUNDS):
                reply_text = ""
                tool_msg: ChatMessage | None = None
                async for ev in provider.stream(history, gen, tool_specs):
                    if isinstance(ev, TextDelta):
                        reply_text += ev.text
                        await self._emit({"type": "delta", "conversation_id": cid, "turn_id": turn.id,
                                          "text": ev.text})
                        if speech_queue is not None:
                            for chunk in chunker.feed(ev.text):
                                speech_queue.put_nowait(chunk)
                    elif isinstance(ev, ToolCallsEvent):
                        tool_msg = ev.message
                if tool_msg is None:
                    break
                # Persist the tool-call turn, run the tools, loop back to the model.
                row = self.db.add_message(cid, "assistant", tool_msg.content,
                                          tool_calls=[t.to_dict() for t in tool_msg.tool_calls])
                await self._emit({"type": "message", "conversation_id": cid, "message": row})
                history.append(tool_msg)
                for call in tool_msg.tool_calls:
                    await self._emit({"type": "tool_call", "conversation_id": cid, "call": call.to_dict()})
                    result = await self.tools.execute(call)
                    row = self.db.add_message(cid, "tool", result.content, tool_call_id=call.id,
                                              name=call.name, meta={"is_error": result.is_error})
                    await self._emit({"type": "message", "conversation_id": cid, "message": row})
                    history.append(result)
                if speech_queue is not None:
                    for chunk in chunker.flush():
                        speech_queue.put_nowait(chunk)
                    chunker = SentenceChunker()

            final = self.db.add_message(
                cid, "assistant", reply_text,
                meta={"persona_id": persona.id, "provider": provider_id, "model": gen.model},
            )
            await self._emit({"type": "assistant_done", "conversation_id": cid, "turn_id": turn.id,
                              "message": final})
            if speech_queue is not None:
                for chunk in chunker.flush():
                    speech_queue.put_nowait(chunk)
                speech_queue.put_nowait(None)
            if speech_task is not None:
                await speech_task
        except asyncio.CancelledError:
            turn.cancel.cancel()
            if speech_task is not None:
                speech_task.cancel()
            if reply_text.strip():
                row = self.db.add_message(cid, "assistant", reply_text, meta={"interrupted": True})
                await self._emit({"type": "assistant_done", "conversation_id": cid, "turn_id": turn.id,
                                  "message": row, "interrupted": True})
            else:
                await self._emit({"type": "assistant_done", "conversation_id": cid, "turn_id": turn.id,
                                  "message": None, "interrupted": True})
            raise
        except ProviderError as e:
            turn.cancel.cancel()
            if speech_task is not None:
                speech_task.cancel()
            if reply_text.strip():
                self.db.add_message(cid, "assistant", reply_text, meta={"error": str(e)})
            await self._emit({"type": "error", "conversation_id": cid, "turn_id": turn.id, "message": str(e)})
            await self._emit({"type": "assistant_done", "conversation_id": cid, "turn_id": turn.id,
                              "message": None, "error": str(e)})
        except Exception as e:  # pragma: no cover - unexpected
            log.exception("turn failed")
            turn.cancel.cancel()
            if speech_task is not None:
                speech_task.cancel()
            await self._emit({"type": "error", "conversation_id": cid, "turn_id": turn.id,
                              "message": f"Unexpected error: {e}"})
            await self._emit({"type": "assistant_done", "conversation_id": cid, "turn_id": turn.id,
                              "message": None, "error": str(e)})

    async def _speak(self, turn: Turn, queue: asyncio.Queue, voice: Voice, language: str | None) -> None:
        """Synthesize queued text chunks in order and push audio to the sinks."""
        idx = 0
        async with self._speech_lock:
            await self._emit({"type": "tts_start", "turn_id": turn.id, "voice_id": voice.id})
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None or turn.cancel.cancelled:
                        break
                    t0 = time.perf_counter()
                    async for audio, sr in self.tts.synthesize(chunk, voice, turn.cancel, language):
                        if turn.cancel.cancelled:
                            break
                        segment_id = f"{turn.id}:{idx}"
                        idx += 1
                        self.sink.play(audio, sr, segment_id)
                        await self._emit({
                            "type": "tts_chunk", "turn_id": turn.id, "segment_id": segment_id,
                            "text": chunk, "duration": round(audio.size / sr, 2),
                            "latency": round(time.perf_counter() - t0, 2),
                        })
            except TTSError as e:
                await self._emit({"type": "error", "turn_id": turn.id, "conversation_id": turn.conversation_id,
                                  "message": f"Voice: {e}"})
            finally:
                await self._emit({"type": "tts_done", "turn_id": turn.id})
