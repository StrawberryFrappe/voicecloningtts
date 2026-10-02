"""Owns TTS engines and runs all synthesis on one dedicated worker thread.

A single GPU worker keeps VRAM usage predictable and avoids two models
fighting over the GPU; asyncio callers receive audio through a queue bridge
so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable

import numpy as np

from ..gpu import GpuBusy, GpuCoordinator
from ..voices import Voice, VoiceLibrary
from .base import Cancelled, CancelToken, TTSEngine, TTSError, VoiceContext
from .chatterbox_engine import ChatterboxEngine
from .remote import RemoteXTTSEngine, xtts_python
from .xtts_engine import XTTSEngine

log = logging.getLogger(__name__)


def engine_classes() -> dict[str, type[TTSEngine]]:
    # XTTS can't share an environment with Chatterbox (conflicting
    # `transformers` pins), so prefer the dedicated .venv-xtts worker when the
    # setup script created one; otherwise try it in-process.
    xtts: type[TTSEngine] = RemoteXTTSEngine if xtts_python() else XTTSEngine
    return {ChatterboxEngine.id: ChatterboxEngine, xtts.id: xtts}

_END = object()


class TTSManager:
    def __init__(self, voices: VoiceLibrary, options: Callable[[str], dict] | None = None,
                 device: str | None = None, gpu: GpuCoordinator | None = None):
        self.voices = voices
        self.gpu = gpu
        self._options = options or (lambda engine_id: {})
        self._device = None if device in (None, "auto") else device
        self._engines: dict[str, TTSEngine] = {}
        self._classes: dict[str, type[TTSEngine]] = engine_classes()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-worker")
        self._status: dict[str, str] = {}  # engine_id -> "loading" | "ready" | "error: ..."

    # -- engines ---------------------------------------------------------
    def register_engine(self, engine: TTSEngine) -> None:
        """Register an engine instance (tests / additional engines)."""
        self._classes[engine.id] = type(engine)
        self._engines[engine.id] = engine

    def engine(self, engine_id: str) -> TTSEngine:
        if engine_id in self._engines:
            eng = self._engines[engine_id]
            eng.options.update(self._options(engine_id))
            return eng
        cls = self._classes.get(engine_id)
        if cls is None:
            raise TTSError(f"Unknown TTS engine '{engine_id}'")
        ok, reason = cls.availability()
        if not ok:
            raise TTSError(reason)
        eng = cls(device=self._device, options=self._options(engine_id))
        self._engines[engine_id] = eng
        return eng

    def set_device(self, device: str | None) -> None:
        self._device = None if device in (None, "auto") else device
        for eng in self._engines.values():
            eng._device = self._device

    def describe(self) -> list[dict]:
        out = []
        for eid, cls in self._classes.items():
            ok, reason = cls.availability()
            eng = self._engines.get(eid)
            info: dict[str, Any] = {
                "id": eid,
                "name": cls.display_name,
                "available": ok,
                "reason": reason,
                "license": cls.license_note,
                "loaded": bool(eng and eng.loaded),
                "status": self._status.get(eid, "idle"),
            }
            if ok:
                probe = eng or cls(device=self._device, options={})
                info["languages"] = probe.languages()
                info["settings"] = [s.to_dict() for s in probe.settings_schema()]
                if eng is not None:
                    info["device"] = eng.device
                    variant = getattr(eng, "loaded_variant", None)
                    if variant:
                        info["variant"] = variant
            out.append(info)
        return out

    # -- voices ----------------------------------------------------------
    def voice_context(self, voice: Voice, language: str | None = None, engine_id: str | None = None) -> VoiceContext:
        engine_id = engine_id or voice.engine
        return VoiceContext(
            voice_id=voice.id,
            reference_path=self.voices.reference_path(voice.id),
            language=language or voice.language,
            cache_dir=lambda key, vid=voice.id: self.voices.cache_dir(vid, key),
            settings=dict(voice.engine_settings.get(engine_id, {})),
        )

    def forget_voice(self, voice_id: str) -> None:
        for eng in self._engines.values():
            forget = getattr(eng, "forget_voice", None)
            if forget:
                forget(voice_id)

    # -- GPU budget ------------------------------------------------------
    async def _claim_gpu(self) -> None:
        """Take the GPU for TTS (low-VRAM mode evicts the voice changer / Whisper)."""
        if self.gpu is None or "tts" in self.gpu.holders and not self.gpu.exclusive:
            return
        try:
            await asyncio.to_thread(self.gpu.acquire, "tts")
        except GpuBusy as e:
            raise TTSError(str(e)) from e

    def _single_engine(self, keep: TTSEngine) -> None:
        """Low-VRAM mode: only one TTS engine resident (runs on the worker thread)."""
        if self.gpu is None or not self.gpu.low_vram:
            return
        for eng in self._engines.values():
            if eng is not keep and eng.loaded:
                log.info("low-VRAM: unloading %s to make room for %s", eng.id, keep.id)
                eng.unload()
                self._status[eng.id] = "idle"

    def _unload_all_now(self) -> None:
        for eng in self._engines.values():
            if eng.loaded:
                eng.unload()
            self._status[eng.id] = "idle"

    def unload_all(self) -> None:
        """Free all TTS VRAM; waits for in-flight synthesis (serialized on the worker)."""
        if threading.current_thread().name.startswith("tts-worker"):
            self._unload_all_now()
        else:
            self._executor.submit(self._unload_all_now).result(timeout=300)

    # -- worker helpers --------------------------------------------------
    async def run(self, fn: Callable, *args) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def load_engine(self, engine_id: str, variant: str | None = None) -> None:
        eng = self.engine(engine_id)
        await self._claim_gpu()
        self._status[engine_id] = "loading"
        try:
            await self.run(lambda: (self._single_engine(eng), eng.load(variant)))
            self._status[engine_id] = "ready"
        except Exception as e:
            self._status[engine_id] = f"error: {e}"
            raise

    async def unload_engine(self, engine_id: str) -> None:
        eng = self._engines.get(engine_id)
        if eng is not None:
            await self.run(eng.unload)
        self._status[engine_id] = "idle"

    async def prepare_voice(self, voice: Voice, language: str | None = None) -> None:
        eng = self.engine(voice.engine)
        ctx = self.voice_context(voice, language)
        await self._claim_gpu()
        self._status[voice.engine] = "loading"
        try:
            await self.run(lambda: (self._single_engine(eng), eng.prepare_voice(ctx)))
            self._status[voice.engine] = "ready"
        except Exception as e:
            self._status[voice.engine] = f"error: {e}"
            raise

    async def synthesize(
        self,
        text: str,
        voice: Voice,
        cancel: CancelToken,
        language: str | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Stream (audio, sample_rate) chunks for ``text`` from the worker thread."""
        eng = self.engine(voice.engine)
        ctx = self.voice_context(voice, language)
        await self._claim_gpu()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def put(item: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def work() -> None:
            if cancel.cancelled:
                put(_END)
                return
            try:
                self._single_engine(eng)
                self._status[eng.id] = "busy"
                for audio, sr in eng.synthesize_stream(text, ctx, cancel):
                    if cancel.cancelled:
                        break
                    put((audio, sr))
                self._status[eng.id] = "ready"
            except Cancelled:
                self._status[eng.id] = "ready"
            except Exception as e:
                log.exception("synthesis failed")
                self._status[eng.id] = f"error: {e}"
                put(e if isinstance(e, TTSError) else TTSError(str(e)))
            finally:
                put(_END)

        loop.run_in_executor(self._executor, work)
        finished = False
        try:
            while True:
                item = await queue.get()
                if item is _END:
                    finished = True
                    break
                if isinstance(item, Exception):
                    finished = True
                    raise item
                yield item
        finally:
            if not finished:
                # Consumer went away mid-stream (stop button / disconnect):
                # make the worker bail out at its next checkpoint.
                cancel.cancel()

    def shutdown(self) -> None:
        for eng in self._engines.values():
            try:
                eng.unload()
            except Exception:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)
