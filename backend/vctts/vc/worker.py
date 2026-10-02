"""Voice-conversion worker process (runs in .venv-vc next to Seed-VC).

JSON-lines protocol on stdin/stdout, one request at a time:

    -> {"id": 1, "op": "load", "device": "auto"}
    -> {"id": 2, "op": "configure", "sample_rate": 48000, "reference_path": "...", "settings": {...}}
    <- {"id": 2, "done": true, "block_frames": 8640, "latency_ms": 380}
    -> {"id": 3, "op": "process", "audio": "<b64 float32>"}
    <- {"id": 3, "done": true, "audio": "<b64 float32>", "infer_ms": 120.5}

Run: ``python -m vctts.vc.worker``. With ``VCTTS_VC_FAKE=1`` a pass-through
converter is used instead of Seed-VC (tests).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..tts.worker import decode_audio, encode_audio
from .seedvc import StreamSettings

log = logging.getLogger("vctts.vc.worker")


class FakeConverter:
    """Pass-through 'conversion' with the same block interface (tests)."""

    def __init__(self, sample_rate: int, settings: StreamSettings):
        zc = sample_rate // 50
        self.block = int(np.round(settings.block_time * sample_rate / zc)) * zc
        self.latency_ms = 1000 * 2 * self.block / sample_rate
        self.last_infer_ms = 0.0
        self.gain = 0.5

    @property
    def block_frames(self) -> int:
        return self.block

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.size != self.block:
            raise ValueError(f"expected {self.block} frames, got {block.size}")
        t0 = time.perf_counter()
        out = (block * self.gain).astype(np.float32)
        self.last_infer_ms = 1000 * (time.perf_counter() - t0)
        return out


class VCWorker:
    def __init__(self, out):
        self.out = out
        self.fake = os.environ.get("VCTTS_VC_FAKE") == "1"
        self.root = Path(os.environ.get("VCTTS_SEEDVC_DIR") or Path.cwd())
        self.model = None
        self.conv = None

    def send(self, msg: dict) -> None:
        self.out.write(json.dumps(msg) + "\n")
        self.out.flush()

    def handle(self, msg: dict) -> dict:
        op = msg.get("op")
        if op == "ping":
            return {"loaded": self.model is not None or self.fake}
        if op == "load":
            if self.fake:
                return {"device": "cpu"}
            if self.model is None:
                from .seedvc import SeedVCModel

                self.model = SeedVCModel(self.root, device=msg.get("device"))
                # VCTTS_VC_RANDOM_INIT=1: build the real architecture without
                # downloading weights (offline/CI checks of the full pipeline).
                self.model.load(random_init=bool(msg.get("random_init"))
                                or os.environ.get("VCTTS_VC_RANDOM_INIT") == "1")
            return {"device": str(self.model.device)}
        if op == "configure":
            settings = StreamSettings.from_dict(msg.get("settings"))
            sr = int(msg["sample_rate"])
            if self.fake:
                self.conv = FakeConverter(sr, settings)
            else:
                if self.model is None:
                    raise RuntimeError("model not loaded")
                from .seedvc import StreamingConverter

                self.model.set_reference(msg["reference_path"], settings.max_prompt_length)
                self.conv = StreamingConverter(self.model, sr, settings)
            return {"block_frames": self.conv.block_frames, "latency_ms": round(self.conv.latency_ms, 1)}
        if op == "process":
            if self.conv is None:
                raise RuntimeError("not configured")
            out = self.conv.process(decode_audio(msg["audio"]))
            return {"audio": encode_audio(out), "infer_ms": round(self.conv.last_infer_ms, 1)}
        if op == "unload":
            self.model = None
            self.conv = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {}
        raise RuntimeError(f"unknown op {op}")

    def run(self, stdin) -> None:
        self.send({"ready": True, "fake": self.fake})
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = msg.get("id")
            try:
                resp = self.handle(msg)
                self.send({"id": rid, "done": True, **resp})
            except Exception as e:
                log.exception("vc op %s failed", msg.get("op"))
                self.send({"id": rid, "error": str(e)})


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")  # Seed-VC's diffusion loop prints a bar per chunk
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s vc-worker: %(message)s")
    proto_out = sys.stdout
    sys.stdout = sys.stderr  # keep the protocol channel clean
    VCWorker(proto_out).run(sys.stdin)


if __name__ == "__main__":
    main()
