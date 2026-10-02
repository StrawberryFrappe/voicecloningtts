"""Fine-tuning jobs: runs vctts.vc.train in the voice-changer environment.

One job at a time. The GPU is taken exclusively for the duration (TTS and the
voice changer are unloaded and refused until it finishes), progress is
published as ``vc_train`` events, and the result is stored with the voice.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from ..gpu import GpuCoordinator
from ..voices import VoiceLibrary
from .changer import VCError, availability, spawn_vc_process

log = logging.getLogger(__name__)

MIN_STEPS, MAX_STEPS = 50, 3000


class TrainingManager:
    def __init__(self, voices: VoiceLibrary, gpu: GpuCoordinator,
                 on_event: Callable[[dict], None] | None = None,
                 precision: Callable[[], str] | None = None,
                 device: Callable[[], str] | None = None,
                 python: Path | None = None, module: str = "vctts.vc.train",
                 extra_args: list[str] | None = None, extra_env: dict | None = None):
        self.voices = voices
        self.gpu = gpu
        self._on_event = on_event or (lambda e: None)
        self._precision = precision or (lambda: "fp32")
        self._device = device or (lambda: "auto")
        self._python = python
        self._module = module
        self._extra_args = extra_args or []
        self._extra_env = extra_env or {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._state: dict = {"state": "idle"}

    # -- info --------------------------------------------------------------
    def defaults(self) -> dict:
        low = self.gpu.low_vram
        return {"steps": 200 if low else 500, "batch_size": 1 if low else 2,
                "min_steps": MIN_STEPS, "max_steps": MAX_STEPS}

    def status(self) -> dict:
        return {**self._state, "defaults": self.defaults()}

    @property
    def running(self) -> bool:
        return self._state.get("state") in ("starting", "running")

    def _update(self, **kw) -> None:
        self._state.update(kw)
        try:
            self._on_event({"type": "vc_train", **self._state})
        except Exception:
            pass

    # -- control -------------------------------------------------------------
    def start(self, voice_id: str, steps: int | None = None, batch_size: int | None = None) -> dict:
        ok, reason = availability()
        if not ok:
            raise VCError(reason)
        voice = self.voices.get(voice_id)
        if voice is None:
            raise VCError("Voice not found")
        d = self.defaults()
        steps = int(min(max(steps or d["steps"], MIN_STEPS), MAX_STEPS))
        batch_size = int(min(max(batch_size or d["batch_size"], 1), 8))
        with self._lock:
            if self.running:
                raise VCError("A fine-tune is already running.")
            self.gpu.begin_exclusive("train")
            try:
                clips = self.voices.list_training(voice_id)
                dataset = self.voices.build_dataset(voice_id)
                out = self.voices.finetune_path(voice_id)
                args = ["--dataset-dir", str(dataset), "--out", str(out), "--steps", str(steps),
                        "--batch-size", str(batch_size), "--precision", self._precision(),
                        "--device", _torch_device(self._device()), *self._extra_args]
                self._proc = spawn_vc_process(self._module, args, python=self._python,
                                              extra_env=self._extra_env, stdin=False)
            except Exception:
                self.gpu.release("train")
                raise
            self._state = {"state": "starting", "voice_id": voice_id, "step": 0, "max_steps": steps,
                           "batch_size": batch_size, "loss": None, "eta_s": None, "message": "Starting…",
                           "error": None, "started_at": time.time(),
                           "clips": len(clips), "seconds": round(sum(c["seconds"] for c in clips), 1)}
            self._update()
            threading.Thread(target=self._watch, args=(self._proc,), daemon=True, name="vc-train").start()
        return self.status()

    def cancel(self) -> dict:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._update(state="cancelling", message="Cancelling…")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        return self.status()

    def shutdown(self) -> None:
        self.cancel()

    # -- worker ----------------------------------------------------------------
    def _watch(self, proc: subprocess.Popen) -> None:
        voice_id = self._state["voice_id"]
        err_tail: list[str] = []

        def pump_err():
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                log.info("[train] %s", line)
                err_tail.append(line)
                del err_tail[:-20]

        threading.Thread(target=pump_err, daemon=True).start()
        done = None
        error = None
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = msg.get("event")
            if ev == "status":
                self._update(state="running", message=msg.get("message"))
            elif ev == "progress":
                step, total = int(msg["step"]), int(msg["max_steps"])
                spp = float(msg.get("sec_per_step") or 0)
                self._update(state="running", step=step, max_steps=total, loss=msg.get("loss"),
                             sec_per_step=spp, eta_s=round(spp * (total - step)), message="Training…")
            elif ev == "done":
                done = msg
            elif ev == "error":
                error = msg.get("error")
        code = proc.wait()
        self._proc = None
        self._cleanup_partial(voice_id)
        try:
            if done and code == 0:
                info = {"steps": int(done.get("steps", self._state.get("max_steps", 0))),
                        "clips": self._state.get("clips"), "seconds": self._state.get("seconds"),
                        "trained_at": time.time()}
                self.voices.set_finetune(voice_id, info)
                self._update(state="done", message="Fine-tune finished", eta_s=0, finetune=info)
            elif self._state.get("state") == "cancelling":
                self._update(state="cancelled", message="Cancelled")
            else:
                msg = error or (err_tail[-1] if err_tail else f"trainer exited with code {code}")
                self._update(state="error", error=msg, message="Fine-tune failed")
        finally:
            self.gpu.release("train")

    def _cleanup_partial(self, voice_id: str) -> None:
        vc_dir = self.voices.finetune_path(voice_id).parent
        for p in vc_dir.glob("tmp*.pth"):  # interrupted atomic save
            p.unlink(missing_ok=True)


def _torch_device(pref: str) -> str:
    if pref in ("auto", "", None):
        return "auto"
    return "cuda:0" if pref == "cuda" else pref
