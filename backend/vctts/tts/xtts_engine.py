"""Coqui XTTS-v2 via the maintained ``coqui-tts`` fork.

XTTS-v2 weights are released under the Coqui Public Model License (CPML),
which only permits non-commercial use. The user has to accept it in Settings
before the model is downloaded.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from typing import Any, Iterator

import numpy as np

from .base import Cancelled, CancelToken, SettingSpec, TTSEngine, TTSError, VoiceContext, free_cuda_memory

log = logging.getLogger(__name__)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000
LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "pl": "Polish", "tr": "Turkish", "ru": "Russian", "nl": "Dutch",
    "cs": "Czech", "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian",
    "ko": "Korean", "hi": "Hindi",
}


def _xtts_lang(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang.startswith("zh"):
        return "zh-cn"
    return lang.split("-")[0]


class XTTSEngine(TTSEngine):
    id = "xtts"
    display_name = "XTTS-v2 (Coqui)"
    license_note = "Coqui Public Model License: non-commercial use only."

    def __init__(self, device: str | None = None, options: dict | None = None):
        super().__init__(device, options)
        self._model = None
        self._latents: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.RLock()

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("TTS") is None:
            return False, "coqui-tts is not installed (pip install -e backend[xtts])"
        if importlib.util.find_spec("torch") is None:
            return False, "PyTorch is not installed"
        return True, ""

    def languages(self) -> dict[str, str]:
        return dict(LANGUAGES)

    def settings_schema(self) -> list[SettingSpec]:
        return [
            SettingSpec("temperature", "Temperature", "float", 0.7, 0.05, 1.5, 0.05),
            SettingSpec("speed", "Speed", "float", 1.0, 0.5, 2.0, 0.05),
            SettingSpec("repetition_penalty", "Repetition penalty", "float", 10.0, 1.0, 20.0, 0.5),
            SettingSpec("top_p", "Top-p", "float", 0.85, 0.1, 1.0, 0.05),
            SettingSpec("stream_chunk_size", "Stream chunk size", "int", 20, 10, 150, 5,
                        help="Smaller = audio starts sooner, slightly more overhead."),
        ]

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, variant: str | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.options.get("license_accepted"):
                raise TTSError(
                    "XTTS-v2 requires accepting the Coqui Public Model License (non-commercial). "
                    "Enable it in Settings > Voice engines."
                )
            os.environ["COQUI_TOS_AGREED"] = "1"
            log.info("loading XTTS-v2 on %s", self.device)
            try:
                from TTS.api import TTS

                tts = TTS(MODEL_NAME, progress_bar=False).to(self.device)
                self._model = tts.synthesizer.tts_model
            except Exception as e:
                self._model = None
                raise TTSError(f"Failed to load XTTS-v2: {e}") from e

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model = None
                self._latents.clear()
                free_cuda_memory()

    def _ensure_voice(self, voice: VoiceContext) -> tuple[Any, Any]:
        if voice.voice_id in self._latents:
            return self._latents[voice.voice_id]
        import torch

        cache_file = voice.cache_dir("xtts-v2") / "latents.pt"
        latents = None
        if cache_file.exists():
            try:
                d = torch.load(cache_file, map_location=self.device, weights_only=True)
                latents = (d["gpt_cond_latent"], d["speaker_embedding"])
            except Exception as e:
                log.warning("ignoring stale XTTS cache for %s: %s", voice.voice_id, e)
        if latents is None:
            gpt, spk = self._model.get_conditioning_latents(audio_path=[str(voice.reference_path)])
            latents = (gpt, spk)
            try:
                torch.save({"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu()}, cache_file)
            except Exception as e:
                log.warning("could not cache XTTS latents: %s", e)
        self._latents[voice.voice_id] = latents
        return latents

    def prepare_voice(self, voice: VoiceContext) -> None:
        with self._lock:
            self.load()
            self._ensure_voice(voice)

    def forget_voice(self, voice_id: str) -> None:
        with self._lock:
            self._latents.pop(voice_id, None)

    def synthesize_stream(
        self, text: str, voice: VoiceContext, cancel: CancelToken
    ) -> Iterator[tuple[np.ndarray, int]]:
        s = self.resolve_settings(voice.settings)
        lang = _xtts_lang(voice.language)
        if lang not in LANGUAGES:
            raise TTSError(f"XTTS-v2 does not support language '{voice.language}'")
        with self._lock:
            self.load()
            gpt, spk = self._ensure_voice(voice)
            cancel.check()
            try:
                stream = self._model.inference_stream(
                    text,
                    lang,
                    gpt,
                    spk,
                    stream_chunk_size=int(s["stream_chunk_size"]),
                    temperature=float(s["temperature"]),
                    speed=float(s["speed"]),
                    repetition_penalty=float(s["repetition_penalty"]),
                    top_p=float(s["top_p"]),
                    enable_text_splitting=True,
                )
                for chunk in stream:
                    cancel.check()
                    yield chunk.squeeze().detach().cpu().numpy().astype(np.float32), SAMPLE_RATE
            except (TTSError, Cancelled):
                raise
            except Exception as e:
                raise TTSError(f"XTTS synthesis failed: {e}") from e
