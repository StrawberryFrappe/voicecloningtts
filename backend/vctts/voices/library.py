"""Saved voices on disk: voices/<id>/{meta.json, reference.wav, cache/}."""

from __future__ import annotations

import io
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..storage import new_id
import numpy as np
import soundfile as sf

from .ingest import IngestError, load_mono, make_reference, split_on_silence

META = "meta.json"
REFERENCE = "reference.wav"
TRAINING = "training"
FINETUNE = "vc/ft_model.pth"


class Voice(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    language: str = "en"
    engine: str = "chatterbox"
    # Per-engine knobs, e.g. {"chatterbox": {"exaggeration": 0.5}, "xtts": {"speed": 1.0}}
    engine_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    duration: float = 0.0
    source_filename: str | None = None
    notes: str = ""
    created_at: float = Field(default_factory=time.time)
    # Voice-changer fine-tune: {"steps", "clips", "seconds", "trained_at"} or None.
    vc_finetune: dict[str, Any] | None = None
    vc_use_finetune: bool = True


class VoiceLibrary:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _dir(self, vid: str) -> Path:
        if not vid or any(c in vid for c in "/\\."):
            raise KeyError(vid)
        return self.root / vid

    def reference_path(self, vid: str) -> Path:
        return self._dir(vid) / REFERENCE

    def cache_dir(self, vid: str, engine_key: str) -> Path:
        d = self._dir(vid) / "cache" / engine_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def clear_cache(self, vid: str) -> None:
        shutil.rmtree(self._dir(vid) / "cache", ignore_errors=True)

    def list(self) -> list[Voice]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            meta = d / META
            if meta.exists() and (d / REFERENCE).exists():
                try:
                    out.append(Voice.model_validate_json(meta.read_text("utf-8")))
                except Exception:
                    continue
        return sorted(out, key=lambda v: v.created_at)

    def get(self, vid: str | None) -> Voice | None:
        if not vid:
            return None
        try:
            meta = self._dir(vid) / META
        except KeyError:
            return None
        if not meta.exists():
            return None
        return Voice.model_validate_json(meta.read_text("utf-8"))

    def _write(self, v: Voice) -> None:
        d = self._dir(v.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (META + ".tmp")
        tmp.write_text(v.model_dump_json(indent=2), "utf-8")
        tmp.replace(d / META)

    def create_from_audio(
        self,
        decoded_wav: Path,
        name: str,
        language: str = "en",
        engine: str = "chatterbox",
        start_s: float | None = None,
        end_s: float | None = None,
        source_filename: str | None = None,
        engine_settings: dict | None = None,
    ) -> tuple[Voice, list[str]]:
        v = Voice(
            name=name.strip() or "Unnamed voice",
            language=language,
            engine=engine,
            source_filename=source_filename,
            engine_settings=engine_settings or {},
        )
        d = self._dir(v.id)
        try:
            info = make_reference(decoded_wav, d / REFERENCE, start_s, end_s)
        except IngestError:
            shutil.rmtree(d, ignore_errors=True)
            raise
        v.duration = info["duration"]
        with self._lock:
            self._write(v)
        return v, info["warnings"]

    def update(self, vid: str, **fields: Any) -> Voice:
        with self._lock:
            v = self.get(vid)
            if v is None:
                raise KeyError(vid)
            allowed = {"name", "language", "engine", "engine_settings", "notes", "vc_use_finetune"}
            data = v.model_dump()
            for k, val in fields.items():
                if k in allowed and val is not None:
                    data[k] = val
            nv = Voice.model_validate(data)
            self._write(nv)
            return nv

    def delete(self, vid: str) -> None:
        with self._lock:
            shutil.rmtree(self._dir(vid), ignore_errors=True)

    # -- voice-changer fine-tuning ------------------------------------------
    def training_dir(self, vid: str) -> Path:
        return self._dir(vid) / TRAINING

    def finetune_path(self, vid: str) -> Path:
        return self._dir(vid) / FINETUNE

    def active_finetune(self, vid: str) -> Path | None:
        """Fine-tuned weights the voice changer should use, if any."""
        v = self.get(vid)
        p = self.finetune_path(vid)
        if v and v.vc_finetune and v.vc_use_finetune and p.exists():
            return p
        return None

    def add_training_audio(self, vid: str, decoded_wav: Path) -> list[dict]:
        if self.get(vid) is None:
            raise KeyError(vid)
        audio, sr = load_mono(decoded_wav)
        clips = split_on_silence(audio, sr)
        if not clips:
            raise IngestError("No usable speech found (need at least 3 s of clear speech).")
        d = self.training_dir(vid)
        d.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem.split("_")[1]) for p in d.glob("clip_*.wav") if p.stem.split("_")[1].isdigit()]
        n = max(existing, default=0)
        for clip in clips:
            n += 1
            sf.write(d / f"clip_{n:04d}.wav", clip, sr, subtype="PCM_16")
        return self.list_training(vid)

    def list_training(self, vid: str) -> list[dict]:
        out = [{"name": REFERENCE, "seconds": round(sf.info(self.reference_path(vid)).duration, 2),
                "removable": False}]
        d = self.training_dir(vid)
        for p in sorted(d.glob("clip_*.wav")) if d.exists() else []:
            out.append({"name": p.name, "seconds": round(sf.info(p).duration, 2), "removable": True})
        return out

    def delete_training_clip(self, vid: str, name: str) -> None:
        if not (name.startswith("clip_") and name.endswith(".wav")) or "/" in name or "\\" in name:
            raise KeyError(name)
        (self.training_dir(vid) / name).unlink(missing_ok=True)

    def build_dataset(self, vid: str) -> Path:
        """Fresh folder with the reference + all training clips, for the trainer."""
        ds = self._dir(vid) / "vc" / "dataset"
        shutil.rmtree(ds, ignore_errors=True)
        ds.mkdir(parents=True)
        shutil.copy2(self.reference_path(vid), ds / REFERENCE)
        for p in sorted(self.training_dir(vid).glob("clip_*.wav")) if self.training_dir(vid).exists() else []:
            shutil.copy2(p, ds / p.name)
        return ds

    def set_finetune(self, vid: str, info: dict | None) -> Voice:
        with self._lock:
            v = self.get(vid)
            if v is None:
                raise KeyError(vid)
            if info is None:
                self.finetune_path(vid).unlink(missing_ok=True)
                shutil.rmtree(self._dir(vid) / "vc" / "dataset", ignore_errors=True)
            data = v.model_dump()
            data["vc_finetune"] = info
            if info is not None:
                data["vc_use_finetune"] = True
            nv = Voice.model_validate(data)
            self._write(nv)
            return nv

    # -- sharing ---------------------------------------------------------
    def export_zip(self, vid: str) -> bytes:
        v = self.get(vid)
        if v is None:
            raise KeyError(vid)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(META, v.model_dump_json(indent=2))
            z.write(self.reference_path(vid), REFERENCE)
            td = self.training_dir(vid)
            for p in sorted(td.glob("clip_*.wav")) if td.exists() else []:
                z.write(p, f"{TRAINING}/{p.name}")
            if self.finetune_path(vid).exists():
                z.write(self.finetune_path(vid), FINETUNE)
        return buf.getvalue()

    def import_zip(self, data: bytes) -> Voice:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                meta = json.loads(z.read(META).decode("utf-8"))
                ref = z.read(REFERENCE)
                extras = {n: z.read(n) for n in z.namelist()
                          if n == FINETUNE or (n.startswith(TRAINING + "/") and n.endswith(".wav")
                                               and "/" not in n[len(TRAINING) + 1:]
                                               and n[len(TRAINING) + 1:].startswith("clip_"))}
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError) as e:
            raise IngestError("Not a valid voice package (.zip with meta.json + reference.wav)") from e
        meta.pop("id", None)
        meta["created_at"] = time.time()
        v = Voice.model_validate(meta)
        d = self._dir(v.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / REFERENCE).write_bytes(ref)
        for name, blob in extras.items():
            target = d / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        if not (d / FINETUNE).exists():
            v.vc_finetune = None
        with self._lock:
            self._write(v)
        return v
