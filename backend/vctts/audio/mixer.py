"""Real-time mixer behind the virtual microphone.

    real mic ──► [mic gain]──[duck while TTS speaks]──┐
                                                       ├─► soft clip ─► virtual cable ("CABLE Input")
    TTS audio ─► [tts gain] ───────────────────────────┘        └──► apps record "CABLE Output"
        └──────────────────────────► monitor (your headphones)

The DSP lives in :class:`MixCore` (pure numpy, unit-tested). :class:`AudioEngine`
wires it to PortAudio streams. The output device callback is the master
clock; the mic and the monitor run through small ring buffers that absorb
clock drift between devices.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from . import devices as devmod
from .resample import resample, to_channels
from .sinks import AudioSink

log = logging.getLogger(__name__)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


class RingBuffer:
    """Thread-safe mono float32 FIFO with overflow dropping."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._r = 0
        self._n = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        return self._n

    def clear(self) -> None:
        with self._lock:
            self._r = 0
            self._n = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        with self._lock:
            if x.size >= self._cap:
                x = x[-self._cap:]
            overflow = self._n + x.size - self._cap
            if overflow > 0:  # drop oldest
                self._r = (self._r + overflow) % self._cap
                self._n -= overflow
            w = (self._r + self._n) % self._cap
            first = min(x.size, self._cap - w)
            self._buf[w:w + first] = x[:first]
            if first < x.size:
                self._buf[: x.size - first] = x[first:]
            self._n += x.size

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            take = min(n, self._n)
            first = min(take, self._cap - self._r)
            out[:first] = self._buf[self._r:self._r + first]
            if first < take:
                out[first:take] = self._buf[: take - first]
            self._r = (self._r + take) % self._cap
            self._n -= take
        return out

    def trim_to(self, max_frames: int) -> int:
        """Drop the oldest frames so at most ``max_frames`` remain (keeps latency low)."""
        with self._lock:
            excess = self._n - max_frames
            if excess > 0:
                self._r = (self._r + excess) % self._cap
                self._n -= excess
                return excess
        return 0


class TtsQueue:
    """Queued speech segments at the mixer rate."""

    def __init__(self) -> None:
        self._segments: deque[tuple[str, np.ndarray]] = deque()
        self._pos = 0
        self._lock = threading.Lock()

    def enqueue(self, segment_id: str, audio: np.ndarray) -> None:
        with self._lock:
            self._segments.append((segment_id, np.asarray(audio, np.float32).reshape(-1)))

    def clear(self) -> list[str]:
        with self._lock:
            ids = [s for s, _ in self._segments]
            self._segments.clear()
            self._pos = 0
            return ids

    @property
    def pending_frames(self) -> int:
        with self._lock:
            return sum(a.size for _, a in self._segments) - self._pos

    @property
    def active(self) -> bool:
        return bool(self._segments)

    def read(self, n: int) -> tuple[np.ndarray, bool, list[str]]:
        out = np.zeros(n, dtype=np.float32)
        done: list[str] = []
        filled = 0
        with self._lock:
            active = bool(self._segments)
            while filled < n and self._segments:
                sid, audio = self._segments[0]
                take = min(n - filled, audio.size - self._pos)
                out[filled:filled + take] = audio[self._pos:self._pos + take]
                filled += take
                self._pos += take
                if self._pos >= audio.size:
                    self._segments.popleft()
                    self._pos = 0
                    done.append(sid)
        return out, active, done


@dataclass
class MixerSettings:
    mic_enabled: bool = True
    mic_gain_db: float = 0.0
    tts_gain_db: float = 0.0
    duck_db: float = -18.0  # mic attenuation while TTS speaks; 0 = off, <= -60 = mute
    monitor_tts: bool = True
    monitor_mic: bool = False
    monitor_gain_db: float = 0.0
    duck_release_ms: float = 250.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "MixerSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, type(getattr(s, k))(v))
        return s

    def to_dict(self) -> dict:
        return asdict(self)


class MixCore:
    def __init__(self, sample_rate: int, settings: MixerSettings | None = None):
        self.sr = sample_rate
        self.settings = settings or MixerSettings()
        self.tts = TtsQueue()
        self._duck = 1.0
        self._hold = 0  # frames to keep ducking after TTS stops
        self.levels = {"mic": 0.0, "tts": 0.0, "out": 0.0}

    def process(self, mic: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
        s = self.settings
        tts, active, done = self.tts.read(n)
        if active:
            self._hold = int(self.sr * s.duck_release_ms / 1000)
        else:
            self._hold = max(0, self._hold - n)
        ducking = (active or self._hold > 0) and s.duck_db < 0
        if not ducking:
            target = 1.0
        elif s.duck_db <= -60:
            target = 0.0
        else:
            target = db_to_gain(s.duck_db)
        ramp = np.linspace(self._duck, target, n, dtype=np.float32)
        self._duck = target

        if mic is None or not s.mic_enabled:
            mic_sig = np.zeros(n, np.float32)
        else:
            mic_sig = mic[:n] * db_to_gain(s.mic_gain_db) * ramp
        tts_sig = tts * db_to_gain(s.tts_gain_db)
        out = soft_clip(mic_sig + tts_sig)

        mon = np.zeros(n, np.float32)
        if s.monitor_tts:
            mon += tts_sig
        if s.monitor_mic:
            mon += mic_sig
        mon = soft_clip(mon * db_to_gain(s.monitor_gain_db))

        decay = 0.85
        for key, sig in (("mic", mic_sig), ("tts", tts_sig), ("out", out)):
            peak = float(np.max(np.abs(sig))) if sig.size else 0.0
            self.levels[key] = max(peak, self.levels[key] * decay)
        return out, mon, done, active


def sanitize(x: np.ndarray) -> np.ndarray:
    """Guard against NaN/inf or out-of-range samples from misbehaving drivers."""
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32, copy=False)


def soft_clip(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Transparent below ``threshold``, smoothly saturating above (no hard clicks)."""
    y = x.astype(np.float32, copy=True)
    over = np.abs(y) > threshold
    if np.any(over):
        sign = np.sign(y[over])
        excess = np.abs(y[over]) - threshold
        y[over] = sign * (threshold + (1 - threshold) * np.tanh(excess / (1 - threshold)))
    return y


@dataclass
class EngineConfig:
    input_device: str | None = None     # real microphone (name)
    output_device: str | None = None    # virtual cable playback endpoint (name)
    monitor_device: str | None = None   # headphones (name)
    sample_rate: int = 48000
    block_ms: int = 10

    @classmethod
    def from_dict(cls, d: dict | None) -> "EngineConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c

    def to_dict(self) -> dict:
        return asdict(self)


class AudioEngine(AudioSink):
    """Virtual-mic sink: mixes TTS with the real mic into the virtual cable."""

    name = "virtual_mic"

    def __init__(self, on_event: Callable[[dict], None] | None = None):
        self.config = EngineConfig()
        self.settings = MixerSettings()
        self._on_event = on_event or (lambda e: None)
        self._core: MixCore | None = None
        self._mic_ring: RingBuffer | None = None
        self._mon_ring: RingBuffer | None = None
        self._streams: list = []
        self._running = False
        self._lock = threading.RLock()
        self._was_active = False
        self.last_error: str | None = None
        self._status_flags = 0

    # -- AudioSink -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._running

    def play(self, audio: np.ndarray, sample_rate: int, segment_id: str) -> None:
        core = self._core
        if core is None:
            return
        core.tts.enqueue(segment_id, resample(audio, sample_rate, core.sr))

    def stop(self) -> None:
        core = self._core
        if core is not None:
            core.tts.clear()

    def pending_seconds(self) -> float:
        core = self._core
        return core.tts.pending_frames / core.sr if core else 0.0

    # -- configuration ---------------------------------------------------
    def update_settings(self, settings: MixerSettings) -> None:
        self.settings = settings
        if self._core is not None:
            self._core.settings = settings

    @property
    def running(self) -> bool:
        return self._running

    def levels(self) -> dict:
        return dict(self._core.levels) if self._core else {"mic": 0.0, "tts": 0.0, "out": 0.0}

    def status(self) -> dict:
        return {
            "running": self._running,
            "config": self.config.to_dict(),
            "settings": self.settings.to_dict(),
            "levels": self.levels(),
            "speaking": bool(self._core and self._core.tts.active),
            "error": self.last_error,
        }

    # -- lifecycle -------------------------------------------------------
    def start(self, config: EngineConfig | None = None) -> None:
        ok, reason = devmod.available()
        if not ok:
            raise RuntimeError(reason)
        import sounddevice as sd

        with self._lock:
            self.shutdown()
            if config is not None:
                self.config = config
            cfg = self.config
            devs = devmod.list_devices()
            out_dev = devmod.find_by_name(cfg.output_device, "output", devs)
            mon_dev = devmod.find_by_name(cfg.monitor_device, "output", devs)
            in_dev = devmod.find_by_name(cfg.input_device, "input", devs)
            if cfg.output_device and out_dev is None:
                raise RuntimeError(f"Output device not found: {cfg.output_device}")
            if cfg.monitor_device and mon_dev is None:
                raise RuntimeError(f"Monitor device not found: {cfg.monitor_device}")
            if cfg.input_device and in_dev is None:
                raise RuntimeError(f"Microphone not found: {cfg.input_device}")
            if out_dev is None and mon_dev is None:
                raise RuntimeError("Choose a virtual cable output and/or a monitor device first.")
            if out_dev is not None and mon_dev is not None and out_dev.id == mon_dev.id:
                mon_dev = None  # same device; avoid opening it twice

            sr = int(cfg.sample_rate)
            block = max(64, int(sr * cfg.block_ms / 1000))
            self._core = MixCore(sr, self.settings)
            self._mic_ring = RingBuffer(sr)  # 1s
            self._mon_ring = RingBuffer(sr)
            extra = _wasapi_settings(sd)
            master_is_monitor = out_dev is None
            max_mic_backlog = block * 4

            core, mic_ring, mon_ring = self._core, self._mic_ring, self._mon_ring

            def master_cb(outdata, frames, _time, status):
                if status:
                    self._status_flags += 1
                mic = None
                if in_dev is not None:
                    mic_ring.trim_to(max_mic_backlog + frames)
                    mic = mic_ring.read(frames)
                out, mon, done, active = core.process(mic, frames)
                if master_is_monitor:
                    outdata[:] = to_channels(mon, outdata.shape[1])
                else:
                    outdata[:] = to_channels(out, outdata.shape[1])
                    if mon_dev is not None:
                        mon_ring.write(mon)
                for sid in done:
                    self._emit({"type": "segment_done", "segment_id": sid})
                if active != self._was_active:
                    self._was_active = active
                    self._emit({"type": "speaking", "active": active})

            def mic_cb(indata, frames, _time, status):
                mic_ring.write(sanitize(indata[:, 0] if indata.ndim > 1 else indata))

            def monitor_cb(outdata, frames, _time, status):
                mon_ring.trim_to(block * 6 + frames)
                outdata[:] = to_channels(mon_ring.read(frames), outdata.shape[1])

            streams = []
            try:
                if in_dev is not None:
                    streams.append(sd.InputStream(
                        device=in_dev.id, samplerate=sr, blocksize=block, channels=1,
                        dtype="float32", latency="low", callback=mic_cb, extra_settings=extra(in_dev),
                    ))
                master_dev = out_dev or mon_dev
                streams.append(sd.OutputStream(
                    device=master_dev.id, samplerate=sr, blocksize=block,
                    channels=min(2, master_dev.max_output_channels), dtype="float32",
                    latency="low", callback=master_cb, extra_settings=extra(master_dev),
                ))
                if out_dev is not None and mon_dev is not None:
                    streams.append(sd.OutputStream(
                        device=mon_dev.id, samplerate=sr, blocksize=block,
                        channels=min(2, mon_dev.max_output_channels), dtype="float32",
                        latency="low", callback=monitor_cb, extra_settings=extra(mon_dev),
                    ))
                for s in streams:
                    s.start()
            except Exception as e:
                for s in streams:
                    try:
                        s.close()
                    except Exception:
                        pass
                self._core = None
                self.last_error = str(e)
                raise RuntimeError(f"Could not open audio devices: {e}") from e
            self._streams = streams
            self._running = True
            self.last_error = None
            log.info("audio engine started: mic=%s out=%s monitor=%s @%d Hz",
                     in_dev and in_dev.name, out_dev and out_dev.name, mon_dev and mon_dev.name, sr)
            self._emit({"type": "engine", "running": True})

    def shutdown(self) -> None:
        with self._lock:
            for s in self._streams:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
            self._streams = []
            was = self._running
            self._running = False
            if self._core is not None:
                for sid in self._core.tts.clear():
                    self._emit({"type": "segment_done", "segment_id": sid})
            self._core = None
            if was:
                self._emit({"type": "engine", "running": False})

    def _emit(self, event: dict) -> None:
        try:
            self._on_event(event)
        except Exception:  # never raise inside an audio callback
            pass


def _wasapi_settings(sd):
    def make(dev: devmod.DeviceInfo):
        if sys.platform == "win32" and "WASAPI" in dev.hostapi:
            try:
                return sd.WasapiSettings(auto_convert=True)
            except TypeError:  # older sounddevice
                return sd.WasapiSettings()
        return None

    return make


def render_offline(core: MixCore, mic: np.ndarray, block: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Run the mixer over whole buffers (tests / debugging)."""
    outs, mons = [], []
    for i in range(0, mic.size, block):
        chunk = mic[i:i + block]
        if chunk.size < block:
            chunk = np.pad(chunk, (0, block - chunk.size))
        o, m, _, _ = core.process(chunk, block)
        outs.append(o)
        mons.append(m)
    return np.concatenate(outs), np.concatenate(mons)

