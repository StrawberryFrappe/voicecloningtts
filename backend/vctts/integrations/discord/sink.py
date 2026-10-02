"""AudioSink that produces Discord-ready PCM.

Discord voice expects 48 kHz, 16-bit, stereo PCM in 20 ms frames (3840
bytes). This sink converts TTS audio into that format and buffers frames; a
future ``discord.AudioSource`` just pops frames from :meth:`read_frame`.
Pure numpy, so it is testable without discord.py installed.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from ...audio.resample import resample
from ...audio.sinks import AudioSink

DISCORD_SR = 48000
FRAME_SAMPLES = 960  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2 * 2  # stereo, int16
SILENCE = b"\x00" * FRAME_BYTES


class DiscordVoiceSink(AudioSink):
    name = "discord"

    def __init__(self) -> None:
        self._frames: deque[bytes] = deque()
        self._partial = b""
        self._lock = threading.Lock()
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    def play(self, audio: np.ndarray, sample_rate: int, segment_id: str) -> None:
        pcm = resample(audio, sample_rate, DISCORD_SR)
        pcm = np.clip(pcm, -1.0, 1.0)
        stereo = np.repeat((pcm * 32767).astype("<i2")[:, None], 2, axis=1).tobytes()
        with self._lock:
            data = self._partial + stereo
            n = len(data) // FRAME_BYTES
            for i in range(n):
                self._frames.append(data[i * FRAME_BYTES:(i + 1) * FRAME_BYTES])
            rest = data[n * FRAME_BYTES:]
            # Pad the tail of a segment so nothing is left hanging.
            self._partial = b""
            if rest:
                self._frames.append(rest + b"\x00" * (FRAME_BYTES - len(rest)))

    def stop(self) -> None:
        with self._lock:
            self._frames.clear()
            self._partial = b""

    def pending_seconds(self) -> float:
        return len(self._frames) * 0.02

    def read_frame(self) -> bytes:
        """Next 20 ms frame, or silence when idle (what discord.AudioSource.read returns)."""
        with self._lock:
            return self._frames.popleft() if self._frames else SILENCE
