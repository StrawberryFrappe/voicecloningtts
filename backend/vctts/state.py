"""Application container: builds and owns every long-lived service."""

from __future__ import annotations

import asyncio
import logging

from .audio.mixer import AudioEngine, EngineConfig, MixerSettings
from .audio.recorder import Recorder
from .audio.sinks import BrowserSink, SinkRouter
from .config import Paths, get_paths
from .conversation import ConversationService
from .events import EventBus
from .gpu import GpuBusy, GpuCoordinator, detect, resolve_low_vram, resolve_precision
from .llm import ProviderRegistry, ToolRegistry
from .personas import PersonaStore
from .secrets import SecretStore
from .storage import Database
from .stt import WhisperSTT
from .tts import TTSManager
from .vc import StreamSettings, VCError, VoiceChanger
from .vc.changer import preset_settings
from .vc.training import TrainingManager
from .voices import VoiceLibrary

log = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "history_limit": 40,
    "browser_playback": "auto",   # auto = only when the virtual mic isn't running
    "tts_device": "auto",
    "xtts_license_accepted": False,
    "stt_model": "auto",
    "stt_device": "auto",
    "stt_language": "",
    "auto_start_audio": True,
    "record_source": "mic",
    "record_device": "",
    "gpu_memory_mode": "auto",    # auto | low | normal (low = one heavy model on the GPU at a time)
    "gpu_precision": "auto",      # auto | fp16 | fp32 (auto = fp32 on GTX 16-series / older cards)
}


class AppState:
    def __init__(self, paths: Paths | None = None, use_keyring: bool = True):
        self.paths = paths or get_paths()
        self.db = Database(self.paths.db)
        self.secrets = SecretStore(self.paths.root / "secrets.json", use_keyring=use_keyring)
        self.bus = EventBus()
        self.providers = ProviderRegistry(self.secrets)
        self.tools = ToolRegistry()
        self.personas = PersonaStore(self.db)
        self.voices = VoiceLibrary(self.paths.voices)
        self.gpu_info = detect()
        self.gpu = GpuCoordinator(lambda: self.low_vram)
        self.tts = TTSManager(self.voices, options=self._engine_options, device=self.setting("tts_device"),
                              gpu=self.gpu)
        self.audio = AudioEngine(on_event=self._audio_event)
        self.audio.config = EngineConfig.from_dict(self.db.get_setting("audio_config"))
        self.audio.update_settings(MixerSettings.from_dict(self.db.get_setting("mixer_settings")))
        self.vc = VoiceChanger(on_event=self._audio_event)
        stored_vc = self.db.get_setting("vc_settings")
        self.vc.settings = (StreamSettings.from_dict(stored_vc) if stored_vc
                            else preset_settings("low" if self.low_vram else "balanced"))
        self.vc.device = self.setting("tts_device") or "auto"
        self.vc.precision = self.precision
        self.audio.voice_changer = self.vc
        self.recorder = Recorder()
        self.stt = WhisperSTT(self.setting("stt_model"), self.setting("stt_device"),
                              low_vram=lambda: self.low_vram, precision=lambda: self.precision)
        self.training = TrainingManager(self.voices, self.gpu, on_event=self._audio_event,
                                        precision=lambda: self.precision,
                                        device=lambda: self.setting("tts_device") or "auto")
        self._vc_resume: str | None = None
        self.gpu.register("tts", self.tts.unload_all)
        self.gpu.register("vc", self._release_vc)
        self.gpu.register("stt", self.stt.unload)
        self.gpu.listeners.append(self._audio_event)
        self.sinks = SinkRouter()
        self.sinks.add(self.audio)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.browser_sink: BrowserSink | None = None
        self.conversations = ConversationService(
            self.db, self.personas, self.providers, self.tools, self.tts, self.voices, self.sinks, self.bus
        )
        self.conversations.on_speech_done = self._after_speech
        self._bg: list[asyncio.Task] = []

    # -- settings --------------------------------------------------------
    def setting(self, key: str):
        return self.db.get_setting(key, DEFAULT_SETTINGS.get(key))

    @property
    def low_vram(self) -> bool:
        return resolve_low_vram(self.setting("gpu_memory_mode"), self.gpu_info)

    @property
    def precision(self) -> str:
        return resolve_precision(self.setting("gpu_precision"), self.gpu_info)

    def gpu_status(self) -> dict:
        return {**self.gpu_info.to_dict(), "low_vram": self.low_vram, "precision": self.precision,
                "memory_mode": self.setting("gpu_memory_mode"), "precision_pref": self.setting("gpu_precision"),
                "holders": sorted(self.gpu.holders), "exclusive": self.gpu.exclusive}

    def all_settings(self) -> dict:
        return {k: self.setting(k) for k in DEFAULT_SETTINGS}

    def update_settings(self, values: dict) -> dict:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS:
                self.db.set_setting(k, v)
        if "tts_device" in values:
            self.tts.set_device(values["tts_device"])
            self.vc.device = values["tts_device"] or "auto"
        if "gpu_precision" in values:
            self.vc.precision = self.precision
            self.vc.stop(keep_model=False)  # reload with the new precision next time
            self.gpu.release("vc")
            self.stt.unload()
        if "stt_model" in values or "stt_device" in values:
            self.stt.configure(values.get("stt_model"), values.get("stt_device"))
            self.stt.unload()
        return self.all_settings()

    def _engine_options(self, engine_id: str) -> dict:
        if engine_id == "xtts":
            return {"license_accepted": bool(self.setting("xtts_license_accepted"))}
        return {}

    # -- lifecycle -------------------------------------------------------
    async def startup(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.browser_sink = BrowserSink(self.bus.publish, self.loop)
        self.sinks.add(_BrowserGate(self.browser_sink, self))
        self._bg.append(asyncio.create_task(self._levels_task()))
        if self.setting("auto_start_audio") and (self.audio.config.output_device or self.audio.config.monitor_device):
            try:
                await asyncio.to_thread(self.audio.start)
            except Exception as e:
                log.warning("audio engine did not auto-start: %s", e)
                self.audio.last_error = str(e)

    async def shutdown(self) -> None:
        for t in self._bg:
            t.cancel()
        try:
            await self.conversations.stop()
        except Exception:
            pass
        if self.recorder.active:
            self.recorder.cancel()
        self.training.shutdown()
        self.vc.shutdown()
        self.audio.shutdown()
        self.tts.shutdown()
        self.db.close()

    # -- voice changer -----------------------------------------------------
    def start_voice_changer(self, voice_id: str | None = None) -> dict:
        if not self.audio.running:
            raise VCError("Start the virtual mic first (Virtual mic page).")
        if not self.audio.config.input_device:
            raise VCError("Choose your microphone on the Virtual mic page first.")
        vid = voice_id or self.db.get_setting("vc_voice_id") or self.db.get_setting("active_voice_id")
        voice = self.voices.get(vid)
        if voice is None:
            raise VCError("Choose a voice for the voice changer.")
        if self.low_vram and self.conversations.speaking:
            raise VCError("Wait for the voice to finish speaking (low-VRAM mode swaps models).")
        self.claim_gpu("vc")
        self.db.set_setting("vc_voice_id", voice.id)
        self.vc.precision = self.precision
        self.vc.start(self.voices.reference_path(voice.id), voice.id, self.audio.config.sample_rate,
                      checkpoint=self.voices.active_finetune(voice.id))
        self._vc_resume = None
        return self.vc.status()

    def claim_gpu(self, feature: str) -> None:
        try:
            self.gpu.acquire(feature)
        except GpuBusy as e:
            raise VCError(str(e)) from e

    def _release_vc(self) -> None:
        """GPU eviction callback: free the voice changer's VRAM (resume later if it was live)."""
        was_live = self.vc.active
        voice_id = self.vc.voice_id
        self.vc.stop(keep_model=False)
        if was_live and self.gpu.exclusive != "train" and self.training.running is False:
            self._vc_resume = voice_id
            log.info("low-VRAM: voice changer paused for TTS; will resume after speech")

    async def _after_speech(self) -> None:
        """Low-VRAM mode: bring the voice changer back once a spoken reply is done."""
        vid, self._vc_resume = self._vc_resume, None
        if not vid or not self.audio.running or self.training.running:
            return
        try:
            await asyncio.to_thread(self.start_voice_changer, vid)
        except VCError as e:
            log.warning("could not resume voice changer: %s", e)

    def _audio_event(self, event: dict) -> None:
        loop = self.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.bus.publish_nowait, {**event, "source": "audio"})

    async def _levels_task(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if not self.bus._subs:
                continue
            if self.audio.running or self.recorder.active:
                self.bus.publish_nowait({
                    "type": "levels",
                    "audio": self.audio.levels() if self.audio.running else None,
                    "recorder": self.recorder.status() if self.recorder.active else None,
                })


class _BrowserGate(BrowserSink):
    """Browser preview playback, enabled according to the ``browser_playback`` setting."""

    name = "browser"

    def __init__(self, inner: BrowserSink, state: AppState):
        self._inner = inner
        self._state = state

    @property
    def enabled(self) -> bool:
        mode = self._state.setting("browser_playback")
        if mode == "on":
            return True
        if mode == "off":
            return False
        return not self._state.audio.running

    def play(self, audio, sample_rate, segment_id):
        self._inner.play(audio, sample_rate, segment_id)

    def stop(self):
        self._inner.stop()
