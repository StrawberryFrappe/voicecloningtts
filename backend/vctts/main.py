"""FastAPI app: REST + WebSocket API and the built UI."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .audio import devices as devmod
from .audio.mixer import EngineConfig, MixerSettings
from .audio.recorder import RecorderError, loopback_available
from .audio.sinks import wav_bytes
from .config import frontend_dist_dir
from .llm import ProviderError
from .personas import LENGTH_PRESETS, Persona, PersonaOverrides
from .state import AppState
from .storage import new_id
from .stt import MODEL_SIZES, STTError
from .tts import CancelToken, TTSError
from .vc import StreamSettings, VCError
from .voices import IngestError, extract_audio, waveform_peaks
from .voices.ingest import load_mono

log = logging.getLogger("vctts")

ALLOWED_UPLOAD_EXT = {
    ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".opus", ".flac", ".webm", ".mkv", ".mov",
    ".aac", ".wma", ".avi",
}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB (videos)


# ---------------------------------------------------------------------------
# request models


class KeyBody(BaseModel):
    api_key: str


class SendBody(BaseModel):
    text: str = Field(min_length=1)
    persona_id: str | None = None
    overrides: PersonaOverrides | None = None
    speak: bool | None = None


class RegenerateBody(BaseModel):
    overrides: PersonaOverrides | None = None
    speak: bool | None = None


class ConversationBody(BaseModel):
    title: str | None = None
    persona_id: str | None = None


class SpeakBody(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str | None = None
    language: str | None = None
    interrupt: bool = False


class CreateVoiceBody(BaseModel):
    upload_id: str
    name: str
    language: str = "en"
    engine: str = "chatterbox"
    start: float | None = None
    end: float | None = None
    engine_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)


class UpdateVoiceBody(BaseModel):
    name: str | None = None
    language: str | None = None
    engine: str | None = None
    engine_settings: dict[str, dict[str, Any]] | None = None
    notes: str | None = None


class PreviewBody(BaseModel):
    text: str | None = None
    play: bool = False  # also route through the sinks (virtual mic / monitor)


class RecordStartBody(BaseModel):
    source: str = "mic"
    device: str | None = None


class RecordStopBody(BaseModel):
    transcribe: bool = True
    language: str | None = None


# ---------------------------------------------------------------------------


def _allowed_origin(origin: str, host: str | None) -> bool:
    try:
        u = urlparse(origin)
    except ValueError:
        return False
    if u.hostname in ("127.0.0.1", "localhost", "::1"):
        return True
    if host and u.netloc == host:
        return True
    return False


def create_app(state: AppState | None = None) -> FastAPI:
    st_holder: dict[str, AppState] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = state or AppState()
        st_holder["s"] = s
        app.state.vctts = s
        await s.startup()
        try:
            yield
        finally:
            await s.shutdown()

    app = FastAPI(title="VoiceCloningTTS", version=__version__, lifespan=lifespan)

    def S() -> AppState:
        return st_holder["s"]

    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        # The API listens on localhost only; block other websites from driving it.
        origin = request.headers.get("origin")
        if origin and not _allowed_origin(origin, request.headers.get("host")):
            return JSONResponse({"detail": "Forbidden origin"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ProviderError)
    async def _provider_err(_r, e: ProviderError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(TTSError)
    async def _tts_err(_r, e: TTSError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(VCError)
    async def _vc_err(_r, e: VCError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(IngestError)
    async def _ingest_err(_r, e: IngestError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    # -- status / settings ------------------------------------------------
    @app.get("/api/status")
    async def status():
        s = S()
        audio_ok, audio_reason = devmod.available()
        stt_ok, stt_reason = s.stt.availability()
        loop_ok, loop_reason = loopback_available() if audio_ok or sys.platform == "win32" else (False, audio_reason)
        return {
            "version": __version__,
            "platform": sys.platform,
            "data_dir": str(s.paths.root),
            "secrets_backend": s.secrets.backend_name,
            "audio": {"available": audio_ok, "reason": audio_reason, **s.audio.status()},
            "virtual_cable": await asyncio.to_thread(devmod.virtual_cable_status) if audio_ok else
            {"installed": False, "device": None, "reason": audio_reason},
            "stt": {"available": stt_ok, "reason": stt_reason, "device": s.stt.last_device},
            "loopback": {"available": loop_ok, "reason": loop_reason},
            "vc": s.vc.status(),
            "busy": s.conversations.busy(),
        }

    @app.get("/api/settings")
    async def get_settings():
        return S().all_settings()

    @app.put("/api/settings")
    async def put_settings(values: dict = Body(...)):
        return S().update_settings(values)

    # -- providers ----------------------------------------------------------
    @app.get("/api/providers")
    async def providers():
        return S().providers.describe()

    @app.put("/api/providers/{pid}/key")
    async def set_key(pid: str, body: KeyBody):
        s = S()
        if pid not in s.providers.ids():
            raise HTTPException(404, "Unknown provider")
        key = body.api_key.strip()
        if not key:
            raise HTTPException(400, "Empty key")
        s.secrets.set(pid, key)
        s.providers.invalidate(pid)
        return {"ok": True}

    @app.delete("/api/providers/{pid}/key")
    async def delete_key(pid: str):
        s = S()
        s.secrets.delete(pid)
        s.providers.invalidate(pid)
        return {"ok": True}

    @app.get("/api/providers/{pid}/models")
    async def models(pid: str):
        p = S().providers.get(pid)
        ms = await p.list_models()
        return [{"id": m.id, "name": m.name} for m in ms]

    # -- tools (extension point) -------------------------------------------
    @app.get("/api/tools")
    async def tools():
        return S().tools.list()

    # -- personas -----------------------------------------------------------
    @app.get("/api/personas")
    async def personas():
        s = S()
        return {"active_id": s.personas.active_id, "personas": [p.model_dump() for p in s.personas.list()],
                "length_presets": LENGTH_PRESETS}

    @app.post("/api/personas")
    async def create_persona(p: Persona):
        p.id = new_id()
        return S().personas.save(p).model_dump()

    @app.put("/api/personas/{pid}")
    async def update_persona(pid: str, p: Persona):
        s = S()
        if s.personas.get(pid) is None:
            raise HTTPException(404, "Persona not found")
        p.id = pid
        return s.personas.save(p).model_dump()

    @app.delete("/api/personas/{pid}")
    async def delete_persona(pid: str):
        s = S()
        if len(s.personas.list()) <= 1:
            raise HTTPException(400, "Keep at least one persona")
        s.personas.delete(pid)
        return {"ok": True, "active_id": s.personas.active_id}

    @app.post("/api/personas/{pid}/duplicate")
    async def duplicate_persona(pid: str):
        p = S().personas.duplicate(pid)
        if p is None:
            raise HTTPException(404, "Persona not found")
        return p.model_dump()

    @app.post("/api/personas/{pid}/activate")
    async def activate_persona(pid: str):
        s = S()
        if s.personas.get(pid) is None:
            raise HTTPException(404, "Persona not found")
        s.personas.set_active(pid)
        return {"ok": True}

    # -- conversations --------------------------------------------------------
    @app.get("/api/conversations")
    async def conversations():
        return S().db.list_conversations()

    @app.post("/api/conversations")
    async def create_conversation(body: ConversationBody | None = None):
        s = S()
        body = body or ConversationBody()
        return s.db.create_conversation(body.title or "New chat", body.persona_id or s.personas.active_id)

    @app.patch("/api/conversations/{cid}")
    async def update_conversation(cid: str, body: ConversationBody):
        s = S()
        conv = s.db.update_conversation(cid, **body.model_dump(exclude_none=True))
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        return conv

    @app.delete("/api/conversations/{cid}")
    async def delete_conversation(cid: str):
        S().db.delete_conversation(cid)
        return {"ok": True}

    @app.get("/api/conversations/{cid}/messages")
    async def messages(cid: str):
        s = S()
        if s.db.get_conversation(cid) is None:
            raise HTTPException(404, "Conversation not found")
        return s.db.list_messages(cid)

    @app.post("/api/conversations/{cid}/send")
    async def send(cid: str, body: SendBody):
        try:
            turn = await S().conversations.send(cid, body.text, body.persona_id, body.overrides, body.speak)
        except KeyError:
            raise HTTPException(404, "Conversation not found")
        return {"turn_id": turn.id}

    @app.post("/api/conversations/{cid}/regenerate")
    async def regenerate(cid: str, body: RegenerateBody | None = None):
        body = body or RegenerateBody()
        try:
            turn = await S().conversations.regenerate(cid, body.overrides, body.speak)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"turn_id": turn.id}

    @app.post("/api/stop")
    async def stop():
        await S().conversations.stop()
        return {"ok": True}

    @app.post("/api/speak")
    async def speak(body: SpeakBody):
        try:
            turn = await S().conversations.speak_text(body.text, body.voice_id, body.language, body.interrupt)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"turn_id": turn.id}

    # -- voices -------------------------------------------------------------
    @app.get("/api/voices")
    async def voices():
        s = S()
        return {"active_id": s.db.get_setting("active_voice_id"), "voices": [v.model_dump() for v in s.voices.list()]}

    @app.put("/api/voices/active")
    async def set_active_voice(body: dict = Body(...)):
        s = S()
        vid = body.get("voice_id")
        if vid and s.voices.get(vid) is None:
            raise HTTPException(404, "Voice not found")
        s.db.set_setting("active_voice_id", vid)
        return {"ok": True}

    @app.post("/api/voices/upload")
    async def upload(file: UploadFile = File(...)):
        s = S()
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXT:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Use mp3, mp4, wav, m4a, ogg, flac, webm...")
        uid = new_id()
        raw = s.paths.uploads / f"{uid}.src{ext}"  # never the same path as the decoded .wav
        size = 0
        with raw.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    f.close()
                    raw.unlink(missing_ok=True)
                    raise HTTPException(413, "File too large")
                f.write(chunk)
        decoded = s.paths.uploads / f"{uid}.wav"
        try:
            await asyncio.to_thread(extract_audio, raw, decoded)
        finally:
            if raw != decoded:
                raw.unlink(missing_ok=True)
        audio, sr = await asyncio.to_thread(load_mono, decoded)
        _cleanup_uploads(s.paths.uploads, keep=decoded)
        return {
            "upload_id": uid,
            "filename": file.filename,
            "duration": round(audio.size / sr, 2),
            "peaks": waveform_peaks(audio, 1200),
        }

    @app.get("/api/uploads/{uid}/audio")
    async def upload_audio(uid: str):
        p = _upload_path(S(), uid)
        return FileResponse(p, media_type="audio/wav")

    @app.post("/api/voices")
    async def create_voice(body: CreateVoiceBody):
        s = S()
        src = _upload_path(s, body.upload_id)
        voice, warnings = await asyncio.to_thread(
            s.voices.create_from_audio, src, body.name, body.language, body.engine,
            body.start, body.end, None, body.engine_settings,
        )
        if not s.db.get_setting("active_voice_id"):
            s.db.set_setting("active_voice_id", voice.id)
        return {"voice": voice.model_dump(), "warnings": warnings}

    @app.patch("/api/voices/{vid}")
    async def update_voice(vid: str, body: UpdateVoiceBody):
        s = S()
        try:
            v = s.voices.update(vid, **body.model_dump(exclude_none=True))
        except KeyError:
            raise HTTPException(404, "Voice not found")
        if body.language is not None or body.engine is not None:
            s.tts.forget_voice(vid)
        return v.model_dump()

    @app.delete("/api/voices/{vid}")
    async def delete_voice(vid: str):
        s = S()
        s.tts.forget_voice(vid)
        s.voices.delete(vid)
        if s.db.get_setting("active_voice_id") == vid:
            remaining = s.voices.list()
            s.db.set_setting("active_voice_id", remaining[0].id if remaining else None)
        return {"ok": True}

    @app.get("/api/voices/{vid}/reference")
    async def voice_reference(vid: str):
        s = S()
        if s.voices.get(vid) is None:
            raise HTTPException(404, "Voice not found")
        return FileResponse(s.voices.reference_path(vid), media_type="audio/wav")

    @app.post("/api/voices/{vid}/prepare")
    async def prepare_voice(vid: str):
        s = S()
        v = s.voices.get(vid)
        if v is None:
            raise HTTPException(404, "Voice not found")
        await s.tts.prepare_voice(v)
        return {"ok": True}

    @app.post("/api/voices/{vid}/preview")
    async def preview_voice(vid: str, body: PreviewBody | None = None):
        s = S()
        body = body or PreviewBody()
        v = s.voices.get(vid)
        if v is None:
            raise HTTPException(404, "Voice not found")
        text = body.text or _preview_text(v.language)
        chunks, sr = [], 24000
        async for audio, sr in s.tts.synthesize(text, v, CancelToken()):
            chunks.append(audio)
        audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        if body.play:
            s.sinks.play(audio, sr, f"preview:{new_id()}")
        return Response(wav_bytes(audio, sr), media_type="audio/wav")

    @app.get("/api/voices/{vid}/export")
    async def export_voice(vid: str):
        s = S()
        v = s.voices.get(vid)
        if v is None:
            raise HTTPException(404, "Voice not found")
        data = await asyncio.to_thread(s.voices.export_zip, vid)
        safe = "".join(c for c in v.name if c.isalnum() or c in " -_").strip() or "voice"
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{safe}.voice.zip"'})

    @app.post("/api/voices/import")
    async def import_voice(file: UploadFile = File(...)):
        data = await file.read()
        v = await asyncio.to_thread(S().voices.import_zip, data)
        return v.model_dump()

    # -- TTS engines --------------------------------------------------------
    @app.get("/api/tts/engines")
    async def engines():
        return S().tts.describe()

    @app.post("/api/tts/engines/{eid}/load")
    async def load_engine(eid: str, body: dict | None = Body(None)):
        await S().tts.load_engine(eid, (body or {}).get("variant"))
        return {"ok": True}

    @app.post("/api/tts/engines/{eid}/unload")
    async def unload_engine(eid: str):
        await S().tts.unload_engine(eid)
        return {"ok": True}

    # -- audio --------------------------------------------------------------
    @app.get("/api/audio/devices")
    async def audio_devices():
        ok, reason = devmod.available()
        if not ok:
            return {"available": False, "reason": reason, "inputs": [], "outputs": [], "virtual": None}
        devs = await asyncio.to_thread(devmod.list_devices)
        virt = devmod.find_virtual_playback(devs)
        return {
            "available": True,
            "reason": "",
            "inputs": [d.to_dict() for d in devs if d.max_input_channels > 0],
            "outputs": [d.to_dict() for d in devs if d.max_output_channels > 0],
            "virtual": virt.to_dict() if virt else None,
            "virtual_status": devmod.virtual_cable_status(),
        }

    @app.get("/api/audio/status")
    async def audio_status():
        return S().audio.status()

    @app.put("/api/audio/config")
    async def audio_config(body: dict = Body(...)):
        s = S()
        cfg = EngineConfig.from_dict({**s.audio.config.to_dict(), **body})
        s.db.set_setting("audio_config", cfg.to_dict())
        was_running = s.audio.running
        s.audio.config = cfg
        if was_running:
            vc_was_active = s.vc.active
            try:
                await asyncio.to_thread(s.audio.start, cfg)
            except Exception as e:
                raise HTTPException(400, str(e))
            if vc_was_active:
                try:
                    await asyncio.to_thread(s.start_voice_changer, s.vc.voice_id)
                except VCError:
                    pass
        return s.audio.status()

    @app.put("/api/audio/mixer")
    async def audio_mixer(body: dict = Body(...)):
        s = S()
        settings = MixerSettings.from_dict({**s.audio.settings.to_dict(), **body})
        s.db.set_setting("mixer_settings", settings.to_dict())
        s.audio.update_settings(settings)
        return s.audio.status()

    @app.post("/api/audio/start")
    async def audio_start():
        s = S()
        try:
            await asyncio.to_thread(s.audio.start)
        except Exception as e:
            raise HTTPException(400, str(e))
        return s.audio.status()

    @app.post("/api/audio/stop")
    async def audio_stop():
        s = S()
        if s.vc.active:
            await asyncio.to_thread(s.vc.stop)
        await asyncio.to_thread(s.audio.shutdown)
        return s.audio.status()

    @app.post("/api/audio/test-tone")
    async def test_tone():
        s = S()
        sr = 48000
        t = np.arange(int(sr * 0.8)) / sr
        tone = (0.25 * np.sin(2 * np.pi * 440 * t) * np.minimum(1, np.minimum(t, t[::-1]) * 20)).astype(np.float32)
        s.sinks.play(tone, sr, f"tone:{new_id()}")
        return {"ok": True, "sinks": [x.name for x in s.sinks.active()]}

    # -- real-time voice changer ----------------------------------------------
    @app.get("/api/vc")
    async def vc_status():
        s = S()
        return {**s.vc.status(), "saved_voice_id": s.db.get_setting("vc_voice_id")}

    @app.put("/api/vc/settings")
    async def vc_settings(body: dict = Body(...)):
        s = S()
        settings = StreamSettings.from_dict({**s.vc.settings.to_dict(), **(body.get("settings") or {})})
        s.db.set_setting("vc_settings", settings.to_dict())
        if body.get("voice_id"):
            if s.voices.get(body["voice_id"]) is None:
                raise HTTPException(404, "Voice not found")
            s.db.set_setting("vc_voice_id", body["voice_id"])
        was_active = s.vc.active
        s.vc.settings = settings
        if was_active:  # apply by reconfiguring (model stays loaded)
            await asyncio.to_thread(s.start_voice_changer, body.get("voice_id") or s.vc.voice_id)
        return {**s.vc.status(), "saved_voice_id": s.db.get_setting("vc_voice_id")}

    @app.post("/api/vc/start")
    async def vc_start(body: dict | None = Body(None)):
        s = S()
        return await asyncio.to_thread(s.start_voice_changer, (body or {}).get("voice_id"))

    @app.post("/api/vc/stop")
    async def vc_stop():
        s = S()
        await asyncio.to_thread(s.vc.stop)
        return s.vc.status()

    @app.post("/api/vc/unload")
    async def vc_unload():
        s = S()
        await asyncio.to_thread(s.vc.stop, False)
        return s.vc.status()

    # -- recording / speech-to-text -----------------------------------------
    @app.post("/api/record/start")
    async def record_start(body: RecordStartBody):
        s = S()
        try:
            return await asyncio.to_thread(s.recorder.start, body.source, body.device or None)
        except RecorderError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/record/status")
    async def record_status():
        return S().recorder.status()

    @app.post("/api/record/cancel")
    async def record_cancel():
        await asyncio.to_thread(S().recorder.cancel)
        return {"ok": True}

    @app.post("/api/record/stop")
    async def record_stop(body: RecordStopBody | None = None):
        s = S()
        body = body or RecordStopBody()
        try:
            rec = await asyncio.to_thread(s.recorder.stop)
        except RecorderError as e:
            raise HTTPException(400, str(e))
        rid = new_id()
        path = s.paths.recordings / f"{rid}.wav"
        await asyncio.to_thread(sf.write, str(path), rec.audio, rec.sample_rate)
        _cleanup_dir(s.paths.recordings, keep=20)
        result: dict[str, Any] = {"recording_id": rid, "duration": round(rec.duration, 2),
                                  "source": rec.source, "device": rec.device}
        if body.transcribe:
            result.update(await _transcribe(s, rec.audio, rec.sample_rate, body.language))
        return result

    @app.post("/api/stt/transcribe")
    async def stt_transcribe(file: UploadFile = File(...), language: str | None = None):
        """Transcribe an uploaded clip (e.g. recorded in the browser)."""
        s = S()
        tmp_in = s.paths.recordings / f"up_{new_id()}{Path(file.filename or 'clip.webm').suffix or '.webm'}"
        tmp_in.write_bytes(await file.read())
        tmp_wav = tmp_in.with_suffix(".decoded.wav")
        try:
            await asyncio.to_thread(extract_audio, tmp_in, tmp_wav, 16000)
            audio, sr = await asyncio.to_thread(load_mono, tmp_wav)
        finally:
            tmp_in.unlink(missing_ok=True)
            tmp_wav.unlink(missing_ok=True)
        return await _transcribe(s, audio, sr, language)

    @app.get("/api/stt/models")
    async def stt_models():
        return {"models": ["auto", *MODEL_SIZES]}

    # -- integrations -------------------------------------------------------
    @app.get("/api/integrations/discord")
    async def discord_status():
        from .integrations.discord import status as discord_status_fn

        return discord_status_fn(S())

    # -- websocket ----------------------------------------------------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        if origin and not _allowed_origin(origin, websocket.headers.get("host")):
            await websocket.close(code=4403)
            return
        await websocket.accept()
        s = S()
        q = s.bus.subscribe()

        async def pump_in():
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "stop":
                    await s.conversations.stop()

        reader = asyncio.create_task(pump_in())
        try:
            await websocket.send_json({"type": "hello", "version": __version__})
            while True:
                getter = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({getter, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    getter.cancel()
                    break
                await websocket.send_json(getter.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()
            s.bus.unsubscribe(q)

    # -- UI -----------------------------------------------------------------
    dist = frontend_dist_dir()
    if dist is not None:
        assets = dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def no_ui():
            return JSONResponse({"detail": "UI not built. Run `npm run build` in frontend/."})

    return app


async def _transcribe(s: AppState, audio: np.ndarray, sr: int, language: str | None) -> dict:
    lang = language if language is not None else (s.setting("stt_language") or None)
    try:
        return await asyncio.to_thread(s.stt.transcribe, audio, sr, lang)
    except STTError as e:
        raise HTTPException(400, str(e))


def _upload_path(s: AppState, uid: str) -> Path:
    if not uid.isalnum():
        raise HTTPException(400, "Bad upload id")
    p = s.paths.uploads / f"{uid}.wav"
    if not p.exists():
        raise HTTPException(404, "Upload expired; upload the file again")
    return p


def _cleanup_uploads(folder: Path, keep: Path, max_files: int = 10) -> None:
    files = sorted((f for f in folder.glob("*.wav") if f != keep), key=lambda f: f.stat().st_mtime)
    for f in files[:-max_files] if len(files) > max_files else []:
        f.unlink(missing_ok=True)


def _cleanup_dir(folder: Path, keep: int) -> None:
    files = sorted(folder.glob("*.wav"), key=lambda f: f.stat().st_mtime)
    for f in files[:-keep] if len(files) > keep else []:
        f.unlink(missing_ok=True)


def _preview_text(language: str) -> str:
    samples = {
        "es": "Hola, esta es una prueba de mi voz clonada. ¿Cómo suena?",
        "fr": "Bonjour, ceci est un test de ma voix clonée. Comment ça sonne ?",
        "de": "Hallo, das ist ein Test meiner geklonten Stimme. Wie klingt das?",
        "pt": "Olá, este é um teste da minha voz clonada. Como soa?",
        "it": "Ciao, questo è un test della mia voce clonata. Come suona?",
    }
    return samples.get((language or "en")[:2], "Hey there! This is a quick test of my cloned voice. How does it sound?")


def run_server(host: str = "127.0.0.1", port: int | None = None) -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("VCTTS_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = port or int(os.environ.get("VCTTS_PORT", "8765"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")



if __name__ == "__main__":
    run_server()
