"""End-to-end API tests: real FastAPI app, fake LLM + fake TTS engine."""

import base64
import io
import time

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vctts.main import create_app
from vctts.state import AppState

from .conftest import FakeEngine, FakeProvider


@pytest.fixture
def client(paths):
    state = AppState(paths, use_keyring=False)
    state.providers.register_override("fake", FakeProvider())
    engine = FakeEngine()
    state.tts.register_engine(engine)
    app = create_app(state)
    with TestClient(app) as c:
        c.state_obj = state
        c.engine = engine
        yield c


def make_voice(client, speech_mp3) -> dict:
    with open(speech_mp3, "rb") as f:
        r = client.post("/api/voices/upload", files={"file": ("me.mp3", f, "audio/mpeg")})
    assert r.status_code == 200, r.text
    up = r.json()
    assert 5.5 < up["duration"] < 6.5 and len(up["peaks"]) == 1200
    r = client.post("/api/voices", json={"upload_id": up["upload_id"], "name": "Me", "language": "en",
                                         "start": 0, "end": up["duration"]})
    assert r.status_code == 200, r.text
    return r.json()["voice"]


def wait_for(ws, type_, timeout=10):
    seen = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        ev = ws.receive_json()
        seen.append(ev)
        if ev["type"] == type_:
            return ev, seen
    raise AssertionError(f"no {type_} event; saw {[e['type'] for e in seen]}")


def test_status_and_settings(client):
    s = client.get("/api/status").json()
    assert s["version"] and "audio" in s and "stt" in s
    r = client.put("/api/settings", json={"history_limit": 10, "bogus": 1})
    assert r.json()["history_limit"] == 10 and "bogus" not in r.json()


def test_provider_keys_never_returned(client):
    assert client.put("/api/providers/openai/key", json={"api_key": "sk-secret"}).status_code == 200
    provs = {p["id"]: p for p in client.get("/api/providers").json()}
    assert provs["openai"]["configured"] and provs["openai"]["key_source"] == "stored"
    assert "sk-secret" not in client.get("/api/providers").text
    assert client.delete("/api/providers/openai/key").status_code == 200
    assert not {p["id"]: p for p in client.get("/api/providers").json()}["openai"]["configured"]
    assert client.put("/api/providers/nope/key", json={"api_key": "x"}).status_code == 404


def test_personas_crud(client):
    data = client.get("/api/personas").json()
    assert len(data["personas"]) == 2 and "short" in data["length_presets"]
    r = client.post("/api/personas", json={"name": "Pirate", "system_prompt": "Talk like a pirate.",
                                           "length": "custom", "custom_max_tokens": 200, "provider": "fake",
                                           "model": "fake-1"})
    pid = r.json()["id"]
    p = r.json()
    p["system_prompt"] = "Arr."
    assert client.put(f"/api/personas/{pid}", json=p).json()["system_prompt"] == "Arr."
    assert client.post(f"/api/personas/{pid}/activate").status_code == 200
    assert client.get("/api/personas").json()["active_id"] == pid
    assert client.post(f"/api/personas/{pid}/duplicate").json()["name"] == "Pirate (copy)"
    assert client.delete(f"/api/personas/{pid}").status_code == 200


def test_voice_lifecycle(client, speech_mp3):
    v = make_voice(client, speech_mp3)
    voices = client.get("/api/voices").json()
    assert voices["active_id"] == v["id"]
    assert client.get(f"/api/voices/{v['id']}/reference").headers["content-type"] == "audio/wav"
    r = client.patch(f"/api/voices/{v['id']}", json={"name": "Renamed", "engine_settings": {"chatterbox": {"speed": 1.5}}})
    assert r.json()["name"] == "Renamed"
    r = client.post(f"/api/voices/{v['id']}/preview", json={"text": "Testing one two."})
    audio, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and audio.size > 0
    assert client.engine.spoken[-1] == "Testing one two."
    assert client.post(f"/api/voices/{v['id']}/prepare").status_code == 200
    z = client.get(f"/api/voices/{v['id']}/export")
    assert z.headers["content-type"] == "application/zip"
    imp = client.post("/api/voices/import", files={"file": ("v.zip", z.content, "application/zip")}).json()
    assert imp["name"] == "Renamed" and imp["id"] != v["id"]
    assert client.delete(f"/api/voices/{v['id']}").status_code == 200
    assert client.get("/api/voices").json()["active_id"] == imp["id"]


def test_upload_rejects_bad_type(client):
    r = client.post("/api/voices/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_chat_over_websocket_with_speech(client, speech_mp3):
    make_voice(client, speech_mp3)
    pid = client.post("/api/personas", json={"name": "T", "provider": "fake", "model": "fake-1"}).json()["id"]
    conv = client.post("/api/conversations", json={"persona_id": pid}).json()
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        r = client.post(f"/api/conversations/{conv['id']}/send", json={"text": "hello"})
        assert r.status_code == 200
        done, seen = wait_for(ws, "assistant_done")
        assert done["message"]["content"] == "Hello there. This is a test reply! Bye."
        deltas = "".join(e["text"] for e in seen if e["type"] == "delta")
        assert deltas == done["message"]["content"]
        # Virtual mic isn't running -> browser preview sink receives WAV audio
        audio_ev, _ = wait_for(ws, "audio")
        wav = base64.b64decode(audio_ev["wav_b64"])
        assert wav[:4] == b"RIFF"
        wait_for(ws, "tts_done")
    msgs = client.get(f"/api/conversations/{conv['id']}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    convs = client.get("/api/conversations").json()
    assert convs[0]["title"] == "hello"


def test_speak_endpoint_and_stop(client, speech_mp3):
    make_voice(client, speech_mp3)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.post("/api/speak", json={"text": "This goes straight to the voice without any model."}).status_code == 200
        wait_for(ws, "tts_done")
        assert client.post("/api/stop").status_code == 200
        wait_for(ws, "stopped")


def test_speak_without_voice_is_400(client):
    assert client.post("/api/speak", json={"text": "hi"}).status_code == 400


def test_origin_guard(client):
    r = client.post("/api/stop", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.post("/api/stop", headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200


def test_audio_and_recording_degrade_gracefully(client):
    devs = client.get("/api/audio/devices").json()
    assert "inputs" in devs
    if not devs["available"]:
        assert client.post("/api/audio/start").status_code == 400
        assert client.post("/api/record/start", json={"source": "mic"}).status_code == 400
    r = client.put("/api/audio/mixer", json={"duck_db": -12, "mic_enabled": False})
    assert r.json()["settings"]["duck_db"] == -12 and r.json()["settings"]["mic_enabled"] is False
    assert client.post("/api/audio/test-tone").status_code == 200


def test_discord_scaffold_status(client):
    d = client.get("/api/integrations/discord").json()
    assert d["enabled"] is False and "token_configured" in d


def test_tools_endpoint_empty(client):
    assert client.get("/api/tools").json() == []
