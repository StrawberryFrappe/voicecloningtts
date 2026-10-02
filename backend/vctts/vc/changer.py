"""Real-time voice changer: live mic → Seed-VC (worker process) → virtual mic.

    mic callback ──feed()──► in_ring ──► processing thread ──► VC worker (GPU)
                                                                   │
    output callback ◄──read()── out_ring ◄─────────────────────────┘

The audio callbacks only touch ring buffers; all model work happens on the
processing thread and in the worker process, so a slow chunk can never stall
the audio devices. When inference falls behind, the oldest input is dropped
to keep latency bounded instead of letting it grow.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio.mixer import RingBuffer
from ..tts.worker import decode_audio, encode_audio
from .seedvc import StreamSettings

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"


def seedvc_root() -> Path:
    return Path(os.environ.get("VCTTS_SEEDVC_DIR") or REPO_ROOT / "vendor" / "seed-vc")


def vc_python() -> Path | None:
    env = os.environ.get("VCTTS_VC_PYTHON")
    candidates = [Path(env)] if env else []
    venv = REPO_ROOT / ".venv-vc"
    candidates += [venv / "Scripts" / "python.exe", venv / "bin" / "python"]
    for c in candidates:
        if c.exists():
            return c
    return None


def availability() -> tuple[bool, str]:
    if os.environ.get("VCTTS_VC_FAKE") == "1":
        return True, ""
    if vc_python() is None:
        return False, "Voice changer not installed (.venv-vc); run scripts/setup_windows.ps1"
    if not (seedvc_root() / "modules").is_dir():
        return False, f"Seed-VC source missing at {seedvc_root()}; run scripts/setup_windows.ps1"
    return True, ""


class VCError(RuntimeError):
    pass


def spawn_vc_process(module: str, args: list[str] | None = None, python: Path | None = None,
                     extra_env: dict | None = None, stdin: bool = True) -> subprocess.Popen:
    """Start ``python -m <module>`` in the voice-changer environment (Seed-VC on path)."""
    py = python or vc_python() or Path(sys.executable)
    root = seedvc_root()
    env = dict(os.environ)
    env.update(extra_env or {})
    env["PYTHONPATH"] = str(REPO_ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["VCTTS_SEEDVC_DIR"] = str(root)
    env.setdefault("TQDM_DISABLE", "1")
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    log.info("starting %s with %s", module, py)
    return subprocess.Popen(
        [str(py), "-m", module, *(args or [])],
        stdin=subprocess.PIPE if stdin else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1, env=env,
        cwd=str(root) if root.is_dir() else None, **kwargs,
    )


class VCWorkerClient:
    """Owns the worker subprocess and does synchronous request/response calls."""

    def __init__(self, python: Path | None = None, extra_env: dict | None = None):
        self._python = python
        self._extra_env = extra_env or {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next = 0

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> subprocess.Popen:
        proc = spawn_vc_process("vctts.vc.worker", python=self._python, extra_env=self._extra_env)
        threading.Thread(target=self._pump_stderr, args=(proc,), daemon=True).start()
        hello = self._read(proc)
        if not hello.get("ready"):
            raise VCError("voice-changer worker failed to start")
        return proc

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            log.info("[vc] %s", line.rstrip())

    @staticmethod
    def _read(proc: subprocess.Popen) -> dict:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                raise VCError(f"voice-changer worker exited (code {proc.poll()}); see the log")
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

    def call(self, op: str, **kw) -> dict:
        with self._lock:
            if not self.alive:
                self._proc = self._start()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            self._next += 1
            rid = self._next
            try:
                proc.stdin.write(json.dumps({"id": rid, "op": op, **kw}) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise VCError("voice-changer worker is not running") from e
            while True:
                resp = self._read(proc)
                if resp.get("id") != rid:
                    continue
                if "error" in resp:
                    raise VCError(resp["error"])
                return resp

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


class VoiceChanger:
    """App-side controller; plugs into :class:`vctts.audio.mixer.AudioEngine`."""

    def __init__(self, on_event: Callable[[dict], None] | None = None, client: VCWorkerClient | None = None):
        self._on_event = on_event or (lambda e: None)
        self.client = client or VCWorkerClient()
        self.settings = StreamSettings()
        self.device = "auto"
        self.precision = "auto"  # "fp16" | "fp32" | "auto" (resolved by vctts.gpu)
        self.checkpoint: Path | None = None  # per-voice fine-tuned weights in use
        self.state = "off"  # off | loading | live | error
        self.error: str | None = None
        self.voice_id: str | None = None
        self._active = False
        self._loaded = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._gen = 0  # run generation; a superseded run never touches shared state
        self._in = RingBuffer(48000 * 4)
        self._out = RingBuffer(48000 * 4)
        self._block = 0
        self._sr = 48000
        self.latency_ms = 0.0
        self._infer_ms: deque[float] = deque(maxlen=50)
        self.dropped_blocks = 0
        self.underruns = 0
        self._lock = threading.Lock()

    # -- audio-thread API (must be cheap and never raise) -----------------
    @property
    def active(self) -> bool:
        return self._active

    def feed(self, mic: np.ndarray) -> None:
        if self._active:
            self._in.write(mic)

    def read(self, frames: int) -> np.ndarray:
        """Converted voice for the output callback (silence while warming up)."""
        if self.state != "live":
            return np.zeros(frames, np.float32)
        if self._out.available < frames:
            self.underruns += 1
        return self._out.read(frames)

    # -- control -----------------------------------------------------------
    def status(self) -> dict:
        ok, reason = availability()
        infer = float(np.mean(self._infer_ms)) if self._infer_ms else 0.0
        block_ms = 1000 * self._block / self._sr if self._block else 1000 * self.settings.block_time
        return {
            "available": ok,
            "reason": reason,
            "state": self.state,
            "active": self._active,
            "error": self.error,
            "voice_id": self.voice_id,
            "finetuned": self.checkpoint is not None,
            "precision": self.precision,
            "model_loaded": self._loaded,
            "settings": self.settings.to_dict(),
            "device": self.device,
            "latency_ms": round(self.latency_ms),
            "infer_ms": round(infer, 1),
            "block_ms": round(block_ms, 1),
            "load": round(infer / block_ms, 2) if block_ms else 0.0,
            "dropped_blocks": self.dropped_blocks,
            "underruns": self.underruns,
        }

    def _emit(self) -> None:
        try:
            self._on_event({"type": "vc", **self.status()})
        except Exception:
            pass

    def start(self, reference_path: Path, voice_id: str, sample_rate: int,
              settings: StreamSettings | None = None, checkpoint: Path | None = None) -> None:
        """Begin converting (model loading happens on a background thread)."""
        ok, reason = availability()
        if not ok:
            raise VCError(reason)
        self.stop(keep_model=True)
        if settings is not None:
            self.settings = settings
        self.voice_id = voice_id
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self._sr = int(sample_rate)
        self._gen += 1
        self._stop = threading.Event()
        self._in.clear()
        self._out.clear()
        self._infer_ms.clear()
        self.dropped_blocks = 0
        self.underruns = 0
        self.error = None
        self.state = "loading"
        self._active = True
        self._emit()
        self._thread = threading.Thread(target=self._run, args=(Path(reference_path), self._stop, self._gen),
                                        daemon=True, name="voice-changer")
        self._thread.start()

    def stop(self, keep_model: bool = True) -> None:
        self._active = False
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5)
        self._thread = None
        if not keep_model:
            self.client.close()
            self._loaded = False
        if self.state != "error":
            self.state = "off"
        self._emit()

    def shutdown(self) -> None:
        self.stop(keep_model=False)

    def _run(self, reference: Path, stop: threading.Event, gen: int) -> None:
        current = lambda: gen == self._gen and not stop.is_set()  # noqa: E731
        try:
            if not self._loaded or not getattr(self.client, "alive", True):
                resp = self.client.call("load", device=self.device, precision=self.precision)
                self._loaded = True
                log.info("voice changer model loaded on %s", resp.get("device"))
            if not current():
                return
            resp = self.client.call("configure", sample_rate=self._sr, reference_path=str(reference),
                                    settings=self.settings.to_dict(),
                                    checkpoint=str(self.checkpoint) if self.checkpoint else None)
            if not current():
                return
            self._block = int(resp["block_frames"])
            self.latency_ms = float(resp["latency_ms"])
            self._in.clear()
            self._out.clear()
            self._out.write(np.zeros(self._block, np.float32))  # one block of jitter headroom
            self.state = "live"
            self._emit()
            last_emit = time.time()
            while current():
                if self._in.available < self._block:
                    time.sleep(0.004)
                    continue
                # Fell behind? drop the oldest audio rather than drifting further back.
                dropped = self._in.trim_to(self._block * 2)
                if dropped:
                    self.dropped_blocks += max(1, dropped // self._block)
                block = self._in.read(self._block)
                resp = self.client.call("process", audio=encode_audio(block))
                self._out.write(decode_audio(resp["audio"]))
                self._out.trim_to(self._block * 3)
                self._infer_ms.append(float(resp.get("infer_ms", 0.0)))
                if time.time() - last_emit > 1.0:
                    last_emit = time.time()
                    self._emit()
        except Exception as e:
            log.exception("voice changer stopped")
            if not getattr(self.client, "alive", True):
                self._loaded = False  # worker died: reload the model next time
            if gen != self._gen:
                return  # superseded run; don't clobber the new one's state
            self.error = str(e)
            self.state = "error"
            self._active = False
            self._emit()


    # -- offline helpers (voice changer must not be live) ------------------
    def _ensure_loaded(self) -> None:
        if self._active:
            raise VCError("Stop the voice changer first.")
        if not self._loaded or not getattr(self.client, "alive", True):
            self.client.call("load", device=self.device, precision=self.precision)
            self._loaded = True

    def benchmark(self, reference_path: Path, sample_rate: int, settings: StreamSettings,
                  checkpoint: Path | None = None, **extra) -> dict:
        """Measure inference time per block for ``settings`` on this machine."""
        self._ensure_loaded()
        return self.client.call("benchmark", sample_rate=sample_rate, reference_path=str(reference_path),
                                settings=settings.to_dict(), checkpoint=str(checkpoint) if checkpoint else None,
                                **extra)

    def convert_file(self, reference_path: Path, input_path: Path, output_path: Path,
                     checkpoint: Path | None = None, settings: StreamSettings | None = None) -> dict:
        self._ensure_loaded()
        return self.client.call("convert_file", reference_path=str(reference_path), input_path=str(input_path),
                                output_path=str(output_path), checkpoint=str(checkpoint) if checkpoint else None,
                                settings=(settings or self.settings).to_dict())


PRESETS: dict[str, dict] = {
    # Ordered best → cheapest; auto-tune walks this list.
    "quality": {"label": "Quality", "block_time": 0.25, "diffusion_steps": 12, "inference_cfg_rate": 0.7,
                "max_prompt_length": 5.0, "extra_time_ce": 2.5, "extra_time": 0.5},
    "balanced": {"label": "Balanced", "block_time": 0.18, "diffusion_steps": 8, "inference_cfg_rate": 0.7,
                 "max_prompt_length": 3.0, "extra_time_ce": 2.5, "extra_time": 0.5},
    "low": {"label": "Low-end GPU", "block_time": 0.30, "diffusion_steps": 4, "inference_cfg_rate": 0.0,
            "max_prompt_length": 3.0, "extra_time_ce": 2.0, "extra_time": 0.5},
    "minimal": {"label": "Minimal", "block_time": 0.45, "diffusion_steps": 3, "inference_cfg_rate": 0.0,
                "max_prompt_length": 2.0, "extra_time_ce": 1.5, "extra_time": 0.5},
}
AUTOTUNE_MAX_LOAD = 0.75


def preset_settings(name: str, base: StreamSettings | None = None) -> StreamSettings:
    values = {k: v for k, v in PRESETS[name].items() if k != "label"}
    return StreamSettings.from_dict({**(base or StreamSettings()).to_dict(), **values})


def autotune(measure: Callable[[StreamSettings], dict], base: StreamSettings | None = None) -> dict:
    """Pick the best preset whose measured load stays under AUTOTUNE_MAX_LOAD."""
    tried = []
    for name in PRESETS:
        settings = preset_settings(name, base)
        result = measure(settings)
        tried.append({"preset": name, **result})
        if result["load"] <= AUTOTUNE_MAX_LOAD:
            return {"preset": name, "settings": settings.to_dict(), "tried": tried, "ok": True}
    # Nothing keeps up: return the cheapest so the user gets the least-bad option.
    return {"preset": name, "settings": settings.to_dict(), "tried": tried, "ok": False}
