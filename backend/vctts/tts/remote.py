"""Proxy engine that runs a real engine in another Python environment.

Used for XTTS-v2: its dependencies conflict with Chatterbox's, so the setup
script installs it into ``.venv-xtts`` and the app drives it through
:mod:`vctts.tts.worker`.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import Cancelled, CancelToken, SettingSpec, TTSEngine, TTSError, VoiceContext
from .worker import decode_audio, voice_to_dict
from .xtts_engine import XTTSEngine

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def xtts_python() -> Path | None:
    """Interpreter of the dedicated XTTS environment, if one is installed."""
    env = os.environ.get("VCTTS_XTTS_PYTHON")
    candidates = [Path(env)] if env else []
    venv = REPO_ROOT / ".venv-xtts"
    candidates += [venv / "Scripts" / "python.exe", venv / "bin" / "python"]
    for c in candidates:
        if c.exists():
            return c
    return None


class RemoteEngine(TTSEngine):
    """Generic proxy; subclasses set ``worker_engine`` and the static metadata."""

    worker_engine = ""

    def __init__(self, device: str | None = None, options: dict | None = None, python: Path | None = None):
        super().__init__(device, options)
        self._python = python
        self._proc: subprocess.Popen | None = None
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._loaded = False
        self.remote_device: str | None = None

    # -- process ---------------------------------------------------------
    def _python_path(self) -> Path:
        p = self._python or xtts_python()
        if p is None:
            raise TTSError("The XTTS environment (.venv-xtts) is not installed. Re-run scripts/setup_windows.ps1.")
        return p

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        py = self._python_path()
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        log.info("starting %s worker with %s", self.worker_engine, py)
        self._proc = subprocess.Popen(
            [str(py), "-m", "vctts.tts.worker", self.worker_engine],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=env, **kwargs,
        )
        threading.Thread(target=self._pump_stderr, args=(self._proc,), daemon=True).start()
        hello = self._read(self._proc)
        if not hello.get("ready"):
            raise TTSError(f"{self.display_name} worker failed to start")
        self._loaded = False
        return self._proc

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            log.info("[%s] %s", self.worker_engine, line.rstrip())

    def _read(self, proc: subprocess.Popen) -> dict:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                code = proc.poll()
                raise TTSError(f"{self.display_name} worker exited unexpectedly (code {code}); see the log")
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _send(self, proc: subprocess.Popen, msg: dict) -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise TTSError(f"{self.display_name} worker is not running") from e

    def _base(self, op: str, **kw) -> dict:
        return {"id": next(self._ids), "op": op, "options": self.options, "device": self._device or "auto", **kw}

    def _call(self, op: str, **kw) -> dict:
        with self._lock:
            proc = self._ensure_proc()
            msg = self._base(op, **kw)
            self._send(proc, msg)
            while True:
                resp = self._read(proc)
                if resp.get("id") != msg["id"]:
                    continue
                if "error" in resp:
                    raise TTSError(resp["error"])
                if resp.get("done"):
                    return resp

    # -- TTSEngine -------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded and self._proc is not None and self._proc.poll() is None

    @property
    def device(self) -> str:
        return self.remote_device or self._device or "auto"

    def load(self, variant: str | None = None) -> None:
        resp = self._call("load", variant=variant)
        self.remote_device = resp.get("device")
        self._loaded = True

    def unload(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            self._loaded = False
            if proc is not None and proc.poll() is None:
                try:
                    if proc.stdin:
                        proc.stdin.close()  # worker exits when stdin closes
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()

    def _voice(self, voice: VoiceContext) -> dict:
        return voice_to_dict(voice, voice.cache_dir(self.cache_key))

    cache_key = "remote"

    def prepare_voice(self, voice: VoiceContext) -> None:
        self._call("prepare", voice=self._voice(voice))
        self._loaded = True

    def forget_voice(self, voice_id: str) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._call("forget", voice_id=voice_id)
            except TTSError:
                pass

    def synthesize_stream(self, text: str, voice: VoiceContext, cancel: CancelToken) -> Iterator[tuple[np.ndarray, int]]:
        with self._lock:
            proc = self._ensure_proc()
            msg = self._base("synth", text=text, voice=self._voice(voice))
            self._send(proc, msg)
            cancel_sent = False
            while True:
                if cancel.cancelled and not cancel_sent:
                    self._send(proc, {"id": msg["id"], "op": "cancel"})
                    cancel_sent = True
                resp = self._read(proc)
                if resp.get("id") != msg["id"]:
                    continue
                if "error" in resp:
                    raise TTSError(resp["error"])
                if resp.get("done"):
                    break
                if "chunk" in resp and not cancel.cancelled:
                    self._loaded = True
                    yield decode_audio(resp["chunk"]), int(resp["sr"])
            if cancel.cancelled:
                raise Cancelled()


class RemoteXTTSEngine(RemoteEngine):
    id = XTTSEngine.id
    display_name = XTTSEngine.display_name
    license_note = XTTSEngine.license_note + " Runs in its own environment (.venv-xtts)."
    worker_engine = "xtts"
    cache_key = "xtts-v2"

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        if xtts_python() is None:
            return False, "XTTS environment not installed (.venv-xtts); run scripts/setup_windows.ps1"
        return True, ""

    def languages(self) -> dict[str, str]:
        return XTTSEngine.languages(self)  # type: ignore[arg-type]

    def settings_schema(self) -> list[SettingSpec]:
        return XTTSEngine.settings_schema(self)  # type: ignore[arg-type]
