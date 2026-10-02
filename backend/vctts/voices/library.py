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
from .ingest import IngestError, make_reference

META = "meta.json"
REFERENCE = "reference.wav"


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
            allowed = {"name", "language", "engine", "engine_settings", "notes"}
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

    # -- sharing ---------------------------------------------------------
    def export_zip(self, vid: str) -> bytes:
        v = self.get(vid)
        if v is None:
            raise KeyError(vid)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(META, v.model_dump_json(indent=2))
            z.write(self.reference_path(vid), REFERENCE)
        return buf.getvalue()

    def import_zip(self, data: bytes) -> Voice:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                meta = json.loads(z.read(META).decode("utf-8"))
                ref = z.read(REFERENCE)
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError) as e:
            raise IngestError("Not a valid voice package (.zip with meta.json + reference.wav)") from e
        meta.pop("id", None)
        meta["created_at"] = time.time()
        v = Voice.model_validate(meta)
        d = self._dir(v.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / REFERENCE).write_bytes(ref)
        with self._lock:
            self._write(v)
        return v
