"""Speech-to-text with faster-whisper (CTranslate2)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from ..audio.resample import resample

log = logging.getLogger(__name__)

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"]
WHISPER_SR = 16000


class STTError(RuntimeError):
    pass


def _add_torch_dll_dirs() -> None:  # pragma: no cover - Windows only
    """Let CTranslate2 find the cuDNN/cuBLAS DLLs that ship with PyTorch."""
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        lib = Path(spec.origin).parent / "lib"
        if lib.is_dir():
            try:
                os.add_dll_directory(str(lib))
                os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass


class WhisperSTT:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        self.model_size = model_size or "auto"
        self.device_pref = device or "auto"
        self._model = None
        self._loaded_key: tuple[str, str] | None = None
        self._lock = threading.Lock()
        self.last_device: str | None = None

    @staticmethod
    def availability() -> tuple[bool, str]:
        if importlib.util.find_spec("faster_whisper") is None:
            return False, "faster-whisper is not installed (pip install -e backend[stt])"
        return True, ""

    def _resolve(self) -> tuple[str, str, str]:
        device = self.device_pref
        if device == "auto":
            device = "cpu"
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                pass
        size = self.model_size
        if size == "auto":
            size = "large-v3-turbo" if device == "cuda" else "small"
        compute = "float16" if device == "cuda" else "int8"
        return size, device, compute

    def configure(self, model_size: str | None = None, device: str | None = None) -> None:
        with self._lock:
            if model_size:
                self.model_size = model_size
            if device:
                self.device_pref = device

    def _load(self):
        size, device, compute = self._resolve()
        if self._model is not None and self._loaded_key == (size, device):
            return self._model
        ok, reason = self.availability()
        if not ok:
            raise STTError(reason)
        _add_torch_dll_dirs()
        from faster_whisper import WhisperModel

        try:
            log.info("loading whisper %s on %s (%s)", size, device, compute)
            self._model = WhisperModel(size, device=device, compute_type=compute)
        except Exception as e:
            if device != "cuda":
                raise STTError(f"Failed to load Whisper: {e}") from e
            log.warning("Whisper on CUDA failed (%s); falling back to CPU", e)
            device, compute = "cpu", "int8"
            self._model = WhisperModel(size, device=device, compute_type=compute)
        self._loaded_key = (size, device)
        self.last_device = device
        return self._model

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._loaded_key = None

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> dict:
        if audio.size == 0:
            return {"text": "", "language": language, "duration": 0.0}
        audio16 = resample(audio.astype(np.float32), sample_rate, WHISPER_SR)
        peak = float(np.max(np.abs(audio16))) if audio16.size else 0.0
        if peak > 0:
            audio16 = audio16 / max(peak, 1e-3) * 0.9  # quiet loopback captures transcribe better normalized
        with self._lock:
            model = self._load()
            try:
                segments, info = model.transcribe(
                    audio16,
                    language=(language or None),
                    vad_filter=True,
                    beam_size=5,
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                raise STTError(f"Transcription failed: {e}") from e
        return {
            "text": text,
            "language": getattr(info, "language", language),
            "duration": round(audio.size / sample_rate, 2),
        }
