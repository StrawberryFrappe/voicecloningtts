from __future__ import annotations

import subprocess
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import pytest
import soundfile as sf

from vctts.config import Paths
from vctts.llm.base import (
    ChatMessage,
    ChatProvider,
    Done,
    GenerationConfig,
    ModelInfo,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallsEvent,
    ToolSpec,
)
from vctts.tts.base import CancelToken, SettingSpec, TTSEngine, VoiceContext
from vctts.voices.ingest import ffmpeg_exe


class FakeProvider(ChatProvider):
    id = "fake"
    display_name = "Fake"
    default_model = "fake-1"

    def __init__(self, script: list[list[StreamEvent]] | None = None, delay: float = 0.0):
        self.script = script or [[TextDelta("Hello there. "), TextDelta("This is a test reply! "),
                                  TextDelta("Bye."), Done("stop")]]
        self.calls: list[tuple[list[ChatMessage], GenerationConfig, list[ToolSpec] | None]] = []
        self.delay = delay

    async def stream(self, messages, config, tools=None) -> AsyncIterator[StreamEvent]:
        import asyncio

        self.calls.append((list(messages), config, tools))
        events = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        for ev in events:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield ev

    async def list_models(self):
        return [ModelInfo("fake-1")]


def tool_call_script() -> list[list[StreamEvent]]:
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    return [
        [TextDelta("Let me add. "), ToolCallsEvent(ChatMessage("assistant", "Let me add. ", [call])), Done("tool_use")],
        [TextDelta("The answer is five."), Done("stop")],
    ]


class FakeEngine(TTSEngine):
    """Deterministic 'voice': a tone whose length follows the text."""

    id = "chatterbox"  # stands in for the default engine
    display_name = "Fake"

    def __init__(self, device=None, options=None):
        super().__init__(device or "cpu", options)
        self._loaded = False
        self.spoken: list[str] = []
        self.prepared: list[str] = []

    @classmethod
    def availability(cls):
        return True, ""

    def languages(self):
        return {"en": "English", "es": "Spanish"}

    def settings_schema(self):
        return [SettingSpec("speed", "Speed", "float", 1.0, 0.5, 2.0, 0.1)]

    @property
    def loaded(self):
        return self._loaded

    def load(self, variant=None):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def prepare_voice(self, voice: VoiceContext):
        self.load()
        assert voice.reference_path.exists()
        self.prepared.append(voice.voice_id)

    def synthesize_stream(self, text, voice: VoiceContext, cancel: CancelToken):
        self.load()
        cancel.check()
        self.spoken.append(text)
        sr = 24000
        n = int(sr * min(0.05 + 0.005 * len(text), 0.5))
        t = np.arange(n) / sr
        yield (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


def make_speechlike(seconds: float = 6.0, sr: int = 24000) -> np.ndarray:
    """A voiced-ish signal with syllable-like amplitude modulation and pauses."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None) ** 0.5
    sig = 0.3 * voiced * env
    sig[: int(0.5 * sr)] = 0  # leading silence (should be trimmed)
    sig[-int(0.5 * sr):] = 0
    return sig.astype(np.float32)


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(tmp_path / "data").ensure()


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    p = tmp_path / "speech.wav"
    sf.write(p, make_speechlike(), 24000)
    return p


@pytest.fixture
def speech_mp3(tmp_path: Path, speech_wav: Path) -> Path:
    out = tmp_path / "speech.mp3"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(speech_wav), str(out)], check=True)
    return out


@pytest.fixture
def speech_mp4(tmp_path: Path, speech_wav: Path) -> Path:
    """A real video file (black frames + AAC audio), like a phone recording."""
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=6",
         "-i", str(speech_wav), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )
    return out
