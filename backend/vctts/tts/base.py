"""TTS engine interface.

Engines are synchronous and GPU-bound; :class:`vctts.tts.manager.TTSManager`
runs them on a dedicated worker thread so the API/UI never block.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


class TTSError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


@dataclass
class VoiceContext:
    voice_id: str
    reference_path: Path
    language: str
    cache_dir: Callable[[str], Path]  # engine_key -> per-voice cache directory
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class SettingSpec:
    key: str
    label: str
    type: str  # "float" | "int" | "choice" | "bool"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    help: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self) -> None:
        if self._ev.is_set():
            raise Cancelled()


class TTSEngine(abc.ABC):
    id: str
    display_name: str
    license_note: str = ""

    def __init__(self, device: str | None = None, options: dict | None = None):
        self._device = device
        self.options = options or {}

    # -- capability ------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def availability(cls) -> tuple[bool, str]:
        """(installed?, reason-if-not). Must not import heavy modules eagerly."""

    @abc.abstractmethod
    def languages(self) -> dict[str, str]:
        ...

    @abc.abstractmethod
    def settings_schema(self) -> list[SettingSpec]:
        ...

    def resolve_settings(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        out = {s.key: s.default for s in self.settings_schema()}
        for k, v in (overrides or {}).items():
            if k in out and v is not None:
                out[k] = v
        return out

    # -- lifecycle -------------------------------------------------------
    @property
    def device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @property
    @abc.abstractmethod
    def loaded(self) -> bool:
        ...

    @abc.abstractmethod
    def load(self, variant: str | None = None) -> None:
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        ...

    # -- synthesis -------------------------------------------------------
    @abc.abstractmethod
    def prepare_voice(self, voice: VoiceContext) -> None:
        """Compute (and cache) speaker conditioning for a voice."""

    @abc.abstractmethod
    def synthesize_stream(
        self, text: str, voice: VoiceContext, cancel: CancelToken
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (mono float32 audio, sample_rate) chunks for ``text``."""

    def synthesize(self, text: str, voice: VoiceContext) -> tuple[np.ndarray, int]:
        chunks, sr = [], 24000
        for audio, sr in self.synthesize_stream(text, voice, CancelToken()):
            chunks.append(audio)
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), sr


def free_cuda_memory() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
