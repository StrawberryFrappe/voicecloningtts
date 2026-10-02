"""Chatterbox (Resemble AI, MIT) zero-shot voice cloning.

Three variants share one engine:

* ``turbo``        – English, lowest latency (default for English voices)
* ``english``      – English, supports emotion exaggeration / CFG
* ``multilingual`` – 23 languages incl. Spanish (default for non-English voices)

Only one variant is resident on the GPU at a time.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Any, Iterator

import numpy as np

from .base import CancelToken, SettingSpec, TTSEngine, TTSError, VoiceContext, free_cuda_memory

log = logging.getLogger(__name__)

MULTILINGUAL_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}
VARIANTS = ("auto", "turbo", "english", "multilingual")


class ChatterboxEngine(TTSEngine):
    id = "chatterbox"
    display_name = "Chatterbox (Resemble AI)"
    license_note = "MIT license. Output carries Resemble's imperceptible Perth watermark."

    def __init__(self, device: str | None = None, options: dict | None = None):
        super().__init__(device, options)
        self._model = None
        self._variant: str | None = None
        self._conds: dict[tuple[str, str], Any] = {}  # (variant, voice_id) -> Conditionals
        self._lock = threading.RLock()

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("chatterbox") is None:
            return False, "chatterbox-tts is not installed (pip install -e backend[chatterbox])"
        if importlib.util.find_spec("torch") is None:
            return False, "PyTorch is not installed"
        return True, ""

    def languages(self) -> dict[str, str]:
        return dict(MULTILINGUAL_LANGUAGES)

    def settings_schema(self) -> list[SettingSpec]:
        return [
            SettingSpec("variant", "Model variant", "choice", "auto", choices=list(VARIANTS),
                        help="auto = Turbo for English voices, Multilingual otherwise."),
            SettingSpec("exaggeration", "Emotion / exaggeration", "float", 0.5, 0.25, 2.0, 0.05,
                        help="Higher = more expressive. Ignored by Turbo."),
            SettingSpec("cfg_weight", "CFG / pace", "float", 0.5, 0.0, 1.0, 0.05,
                        help="Lower = slower, more deliberate delivery. Try 0.3 for fast-talking references."),
            SettingSpec("temperature", "Temperature", "float", 0.8, 0.05, 2.0, 0.05),
        ]

    # -- lifecycle -------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_variant(self) -> str | None:
        return self._variant

    def pick_variant(self, requested: str | None, language: str) -> str:
        lang = (language or "en").lower().split("-")[0]
        if requested and requested != "auto":
            if requested in ("turbo", "english") and lang != "en":
                return "multilingual"
            return requested
        return "turbo" if lang == "en" else "multilingual"

    def load(self, variant: str | None = None) -> None:
        variant = variant or "turbo"
        with self._lock:
            if self._model is not None and self._variant == variant:
                return
            self.unload()
            log.info("loading Chatterbox %s on %s", variant, self.device)
            try:
                if variant == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS as M
                elif variant == "multilingual":
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M
                else:
                    from chatterbox.tts import ChatterboxTTS as M
                self._model = M.from_pretrained(device=self.device)
            except Exception as e:
                self._model = None
                raise TTSError(f"Failed to load Chatterbox ({variant}): {e}") from e
            self._variant = variant

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model = None
                self._variant = None
                self._conds.clear()
                free_cuda_memory()

    # -- voices ----------------------------------------------------------
    def _conditionals_cls(self):
        if self._variant == "turbo":
            from chatterbox.tts_turbo import Conditionals
        elif self._variant == "multilingual":
            from chatterbox.mtl_tts import Conditionals
        else:
            from chatterbox.tts import Conditionals
        return Conditionals

    def _ensure_voice(self, voice: VoiceContext, exaggeration: float) -> None:
        key = (self._variant or "", voice.voice_id)
        if key in self._conds:
            self._model.conds = self._conds[key]
            return
        cache_file = voice.cache_dir(f"chatterbox-{self._variant}") / "conds.pt"
        conds = None
        if cache_file.exists():
            try:
                conds = self._conditionals_cls().load(cache_file, map_location=self.device).to(self.device)
            except Exception as e:  # stale cache from another version
                log.warning("ignoring stale Chatterbox cache for %s: %s", voice.voice_id, e)
        if conds is None:
            try:
                self._model.prepare_conditionals(str(voice.reference_path), exaggeration=exaggeration)
            except AssertionError as e:
                raise TTSError(f"Reference clip rejected by Chatterbox: {e}") from e
            conds = self._model.conds
            try:
                conds.save(cache_file)
            except Exception as e:
                log.warning("could not cache conditionals: %s", e)
        self._conds[key] = conds
        self._model.conds = conds

    def prepare_voice(self, voice: VoiceContext) -> None:
        s = self.resolve_settings(voice.settings)
        with self._lock:
            self.load(self.pick_variant(s["variant"], voice.language))
            self._ensure_voice(voice, float(s["exaggeration"]))

    def forget_voice(self, voice_id: str) -> None:
        with self._lock:
            for k in [k for k in self._conds if k[1] == voice_id]:
                del self._conds[k]

    # -- synthesis -------------------------------------------------------
    def synthesize_stream(
        self, text: str, voice: VoiceContext, cancel: CancelToken
    ) -> Iterator[tuple[np.ndarray, int]]:
        s = self.resolve_settings(voice.settings)
        with self._lock:
            variant = self.pick_variant(s["variant"], voice.language)
            self.load(variant)
            cancel.check()
            self._ensure_voice(voice, float(s["exaggeration"]))
            cancel.check()
            kwargs: dict[str, Any] = {"temperature": float(s["temperature"])}
            if variant != "turbo":
                kwargs["exaggeration"] = float(s["exaggeration"])
                kwargs["cfg_weight"] = float(s["cfg_weight"])
            if variant == "multilingual":
                lang = (voice.language or "en").lower().split("-")[0]
                if lang not in MULTILINGUAL_LANGUAGES:
                    raise TTSError(f"Chatterbox does not support language '{voice.language}'")
                kwargs["language_id"] = lang
            try:
                wav = self._model.generate(text, **kwargs)
            except Exception as e:
                raise TTSError(f"Chatterbox synthesis failed: {e}") from e
            audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
            sr = int(self._model.sr)
        yield audio, sr
