"""One-shot recording for the record → transcribe → edit → send flow.

Two sources:

* ``mic``      – any input device (sounddevice / PortAudio)
* ``loopback`` – what's playing on an output device (your headphones).
  On Windows this uses WASAPI loopback through PyAudioWPatch, because
  PortAudio's stock WASAPI backend can't open loopback endpoints. On Linux a
  PulseAudio/PipeWire ``*.monitor`` source is used.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import devices as devmod
from .mixer import sanitize

log = logging.getLogger(__name__)

MAX_SECONDS = 300


class RecorderError(RuntimeError):
    pass


@dataclass
class Recording:
    audio: np.ndarray  # mono float32
    sample_rate: int
    source: str
    device: str | None

    @property
    def duration(self) -> float:
        return self.audio.size / self.sample_rate if self.sample_rate else 0.0


def loopback_available() -> tuple[bool, str]:
    if sys.platform == "win32":
        if importlib.util.find_spec("pyaudiowpatch") is None:
            return False, "Install PyAudioWPatch for headphone capture (pip install -e backend[loopback])"
        return True, ""
    ok, reason = devmod.available()
    if not ok:
        return False, reason
    if any("monitor" in d.name.lower() for d in devmod.list_devices(all_hostapis=True) if d.max_input_channels):
        return True, ""
    return False, "No loopback/monitor capture device found"


class Recorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._sr = 0
        self._channels = 1
        self._source: str | None = None
        self._device: str | None = None
        self._started = 0.0
        self._level = 0.0
        self._closer = None  # callable that stops/cleans up the active stream

    @property
    def active(self) -> bool:
        return self._closer is not None

    def status(self) -> dict:
        return {
            "recording": self.active,
            "source": self._source,
            "device": self._device,
            "elapsed": round(time.time() - self._started, 1) if self.active else 0.0,
            "level": round(self._level, 3),
        }

    def _on_audio(self, mono: np.ndarray) -> None:
        mono = sanitize(mono)
        with self._lock:
            self._chunks.append(mono.copy())
            total = sum(c.size for c in self._chunks)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        self._level = max(peak, self._level * 0.8)
        if self._sr and total > self._sr * MAX_SECONDS:
            log.warning("recording hit %ss limit", MAX_SECONDS)

    def start(self, source: str = "mic", device: str | None = None) -> dict:
        if self.active:
            raise RecorderError("Already recording")
        with self._lock:
            self._chunks = []
        self._level = 0.0
        if source == "mic":
            self._start_mic(device)
        elif source == "loopback":
            if sys.platform == "win32":
                self._start_wasapi_loopback(device)
            else:
                self._start_monitor_source(device)
        else:
            raise RecorderError(f"Unknown source '{source}'")
        self._source = source
        self._started = time.time()
        return self.status()

    def stop(self) -> Recording:
        if not self.active:
            raise RecorderError("Not recording")
        closer, self._closer = self._closer, None
        try:
            closer()
        except Exception as e:  # pragma: no cover
            log.warning("error closing recorder: %s", e)
        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
            self._chunks = []
        max_frames = int(self._sr * MAX_SECONDS)
        return Recording(audio[:max_frames].astype(np.float32), self._sr, self._source or "mic", self._device)

    def cancel(self) -> None:
        if self.active:
            self.stop()

    # -- sources ---------------------------------------------------------
    def _start_mic(self, device: str | None) -> None:
        ok, reason = devmod.available()
        if not ok:
            raise RecorderError(reason)
        import sounddevice as sd

        dev = devmod.find_by_name(device, "input") if device else devmod.default_device("input")
        if device and dev is None:
            raise RecorderError(f"Microphone not found: {device}")
        sr = int(dev.default_samplerate) if dev else 48000

        def cb(indata, frames, _t, status):
            self._on_audio(indata.mean(axis=1) if indata.ndim > 1 else indata)

        try:
            stream = sd.InputStream(
                device=dev.id if dev else None, samplerate=sr, channels=1, dtype="float32", callback=cb
            )
            stream.start()
        except Exception as e:
            raise RecorderError(f"Could not open microphone: {e}") from e
        self._sr = sr
        self._device = dev.name if dev else "default"
        self._closer = lambda: (stream.stop(), stream.close())

    def _start_monitor_source(self, device: str | None) -> None:
        import sounddevice as sd

        devs = [d for d in devmod.list_devices(all_hostapis=True) if d.max_input_channels > 0]
        cands = [d for d in devs if "monitor" in d.name.lower()]
        if device:
            cands = [d for d in cands if device.lower() in d.name.lower()] or cands
        if not cands:
            raise RecorderError("No monitor source found for loopback capture")
        dev = cands[0]
        sr = int(dev.default_samplerate)

        def cb(indata, frames, _t, status):
            self._on_audio(indata.mean(axis=1))

        stream = sd.InputStream(device=dev.id, samplerate=sr, channels=min(2, dev.max_input_channels),
                                dtype="float32", callback=cb)
        stream.start()
        self._sr = sr
        self._device = dev.name
        self._closer = lambda: (stream.stop(), stream.close())

    def _start_wasapi_loopback(self, device: str | None) -> None:  # pragma: no cover - Windows only
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RecorderError(
                "Headphone capture needs PyAudioWPatch (pip install -e backend[loopback])"
            ) from e
        p = pyaudio.PyAudio()
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            target_name = device
            if not target_name:
                spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
                target_name = spk["name"]
            loop_dev = None
            for d in p.get_loopback_device_info_generator():
                if target_name.lower() in d["name"].lower():
                    loop_dev = d
                    break
            if loop_dev is None:
                raise RecorderError(f"No loopback endpoint for output device '{target_name}'")
            channels = int(loop_dev["maxInputChannels"]) or 2
            sr = int(loop_dev["defaultSampleRate"])

            def cb(in_data, frame_count, time_info, status):
                data = np.frombuffer(in_data, dtype=np.float32)
                if channels > 1:
                    data = data.reshape(-1, channels).mean(axis=1)
                self._on_audio(data)
                return (None, pyaudio.paContinue)

            stream = p.open(
                format=pyaudio.paFloat32, channels=channels, rate=sr, input=True,
                input_device_index=loop_dev["index"], frames_per_buffer=1024, stream_callback=cb,
            )
            stream.start_stream()
        except RecorderError:
            p.terminate()
            raise
        except Exception as e:
            p.terminate()
            raise RecorderError(f"Could not start loopback capture: {e}") from e

        self._sr = sr
        self._device = loop_dev["name"]

        def close():
            stream.stop_stream()
            stream.close()
            p.terminate()

        self._closer = close
