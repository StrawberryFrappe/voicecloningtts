"""Audio sinks: where synthesized speech goes.

Every consumer of TTS audio (the virtual-mic mixer, the browser preview, a
future Discord voice connection, a file recorder in tests) implements
:class:`AudioSink`. The conversation pipeline only talks to a
:class:`SinkRouter`, so adding a new destination never touches the pipeline.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import io
import threading
from typing import Awaitable, Callable

import numpy as np
import soundfile as sf


class AudioSink(abc.ABC):
    name: str = "sink"

    @property
    def enabled(self) -> bool:
        return True

    @abc.abstractmethod
    def play(self, audio: np.ndarray, sample_rate: int, segment_id: str) -> None:
        """Queue mono float32 audio for playback. Must not block."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Drop everything queued and stop the current playback immediately."""

    def pending_seconds(self) -> float:
        return 0.0


class MemorySink(AudioSink):
    """Collects audio in memory (tests, offline rendering)."""

    name = "memory"

    def __init__(self) -> None:
        self.segments: list[tuple[str, np.ndarray, int]] = []
        self.stopped = 0
        self._lock = threading.Lock()

    def play(self, audio: np.ndarray, sample_rate: int, segment_id: str) -> None:
        with self._lock:
            self.segments.append((segment_id, audio, sample_rate))

    def stop(self) -> None:
        with self._lock:
            self.stopped += 1


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class BrowserSink(AudioSink):
    """Sends audio to the UI over the event bus (preview without a virtual cable)."""

    name = "browser"

    def __init__(self, publish: Callable[[dict], Awaitable[None] | None], loop: asyncio.AbstractEventLoop):
        self._publish = publish
        self._loop = loop
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    def play(self, audio: np.ndarray, sample_rate: int, segment_id: str) -> None:
        payload = {
            "type": "audio",
            "segment_id": segment_id,
            "sample_rate": sample_rate,
            "duration": round(audio.size / sample_rate, 3),
            "wav_b64": base64.b64encode(wav_bytes(audio, sample_rate)).decode("ascii"),
        }
        self._emit(payload)

    def stop(self) -> None:
        self._emit({"type": "audio_stop"})

    def _emit(self, payload: dict) -> None:
        def go() -> None:
            res = self._publish(payload)
            if asyncio.iscoroutine(res):
                asyncio.ensure_future(res)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            go()
        else:
            self._loop.call_soon_threadsafe(go)


class SinkRouter(AudioSink):
    """Fans audio out to every enabled sink."""

    name = "router"

    def __init__(self) -> None:
        self._sinks: dict[str, AudioSink] = {}

    def add(self, sink: AudioSink) -> None:
        self._sinks[sink.name] = sink

    def remove(self, name: str) -> None:
        self._sinks.pop(name, None)

    def get(self, name: str) -> AudioSink | None:
        return self._sinks.get(name)

    def active(self) -> list[AudioSink]:
        return [s for s in self._sinks.values() if s.enabled]

    def play(self, audio: np.ndarray, sample_rate: int, segment_id: str) -> None:
        for s in self.active():
            s.play(audio, sample_rate, segment_id)

    def stop(self) -> None:
        for s in self._sinks.values():
            s.stop()

    def pending_seconds(self) -> float:
        return max((s.pending_seconds() for s in self.active()), default=0.0)

    def describe(self) -> list[dict]:
        return [{"name": n, "enabled": s.enabled} for n, s in self._sinks.items()]


def b64_wav(audio: np.ndarray, sample_rate: int) -> str:
    return base64.b64encode(wav_bytes(audio, sample_rate)).decode("ascii")
