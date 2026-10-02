"""Out-of-process TTS worker.

Some engines can't share a Python environment (Chatterbox pins
``transformers`` to versions coqui-tts can't use), so XTTS runs in its own
venv and the app talks to it over a JSON-lines protocol on stdin/stdout:

    -> {"id": 1, "op": "synth", "text": "...", "voice": {...}, "options": {...}}
    <- {"id": 1, "chunk": "<base64 float32 LE>", "sr": 24000}
    <- {"id": 1, "done": true}            (or {"id": 1, "error": "..."})
    -> {"id": 1, "op": "cancel"}          (any time; stops the running synth)

Run: ``python -m vctts.tts.worker xtts``
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .base import Cancelled, CancelToken, TTSEngine, VoiceContext

log = logging.getLogger("vctts.worker")


def encode_audio(audio: np.ndarray) -> str:
    return base64.b64encode(np.asarray(audio, dtype="<f4").tobytes()).decode("ascii")


def decode_audio(data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype="<f4").astype(np.float32)


def voice_to_dict(ctx: VoiceContext, cache_root: Path) -> dict:
    return {
        "voice_id": ctx.voice_id,
        "reference_path": str(ctx.reference_path),
        "language": ctx.language,
        "cache_root": str(cache_root),
        "settings": ctx.settings,
    }


def voice_from_dict(d: dict) -> VoiceContext:
    root = Path(d["cache_root"])

    def cache_dir(_key: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    return VoiceContext(
        voice_id=d["voice_id"],
        reference_path=Path(d["reference_path"]),
        language=d["language"],
        cache_dir=cache_dir,
        settings=d.get("settings") or {},
    )


def make_engine(name: str) -> TTSEngine:
    if ":" in name:  # "package.module:Class" (tests, third-party engines)
        import importlib

        mod, cls = name.split(":", 1)
        return getattr(importlib.import_module(mod), cls)()
    if name == "xtts":
        from .xtts_engine import XTTSEngine

        return XTTSEngine()
    if name == "chatterbox":
        from .chatterbox_engine import ChatterboxEngine

        return ChatterboxEngine()
    raise SystemExit(f"unknown engine {name}")


class Worker:
    def __init__(self, engine: TTSEngine, out):
        self.engine = engine
        self.out = out
        self.out_lock = threading.Lock()
        self.inbox: queue.Queue = queue.Queue()
        self.cancels: dict[Any, CancelToken] = {}

    def send(self, msg: dict) -> None:
        with self.out_lock:
            self.out.write(json.dumps(msg) + "\n")
            self.out.flush()

    def reader(self, stream) -> None:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("op") == "cancel":
                tok = self.cancels.get(msg.get("id"))
                if tok:
                    tok.cancel()
                continue
            if msg.get("id") is not None:
                self.cancels[msg["id"]] = CancelToken()
            self.inbox.put(msg)
        self.inbox.put(None)  # stdin closed -> exit

    def handle(self, msg: dict) -> None:
        rid = msg.get("id")
        op = msg.get("op")
        eng = self.engine
        eng.options.update(msg.get("options") or {})
        if msg.get("device"):
            eng._device = None if msg["device"] == "auto" else msg["device"]
        try:
            if op == "ping":
                self.send({"id": rid, "done": True, "loaded": eng.loaded})
            elif op == "load":
                eng.load(msg.get("variant"))
                self.send({"id": rid, "done": True, "device": eng.device})
            elif op == "unload":
                eng.unload()
                self.send({"id": rid, "done": True})
            elif op == "prepare":
                eng.prepare_voice(voice_from_dict(msg["voice"]))
                self.send({"id": rid, "done": True})
            elif op == "forget":
                forget = getattr(eng, "forget_voice", None)
                if forget:
                    forget(msg["voice_id"])
                self.send({"id": rid, "done": True})
            elif op == "synth":
                tok = self.cancels.get(rid) or CancelToken()
                try:
                    for audio, sr in eng.synthesize_stream(msg["text"], voice_from_dict(msg["voice"]), tok):
                        if tok.cancelled:
                            break
                        self.send({"id": rid, "chunk": encode_audio(audio), "sr": sr})
                except Cancelled:
                    pass
                self.send({"id": rid, "done": True})
            else:
                self.send({"id": rid, "error": f"unknown op {op}"})
        except Exception as e:
            log.exception("worker op %s failed", op)
            self.send({"id": rid, "error": str(e)})
        finally:
            self.cancels.pop(rid, None)

    def run(self, stdin) -> None:
        threading.Thread(target=self.reader, args=(stdin,), daemon=True).start()
        self.send({"ready": True, "engine": self.engine.id})
        while True:
            msg = self.inbox.get()
            if msg is None:
                break
            self.handle(msg)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    name = argv[0] if argv else "xtts"
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s worker: %(message)s")
    # Keep the protocol channel clean: anything libraries print goes to stderr.
    proto_out = sys.stdout
    sys.stdout = sys.stderr
    Worker(make_engine(name), proto_out).run(sys.stdin)


if __name__ == "__main__":
    main()
