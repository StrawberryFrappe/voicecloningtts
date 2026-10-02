import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vctts import doctor as doc
from vctts.doctor import FAIL, OK, SKIP, WARN, Doctor, format_report, hf_cached, probe_env, redact
from vctts.gpu import GpuInfo
from vctts.llm.base import Done, TextDelta
from vctts.main import create_app
from vctts.state import AppState

from .conftest import FakeEngine, FakeProvider


@pytest.fixture
def state(paths):
    s = AppState(paths, use_keyring=False)
    yield s
    s.tts.shutdown()


def by_id(checks, cid):
    return next(c for c in checks if c.id == cid)


async def test_quick_run_is_complete_and_saves_report(state, monkeypatch):
    monkeypatch.setattr(Doctor, "check_network", lambda self: [doc.Check("net:huggingface", "net", OK, "fake")])
    rep = await Doctor(state).run(deep=False)
    ids = [c["id"] for c in rep["checks"]]
    for expected in ("system", "disk", "ui", "ffmpeg", "gpu", "env_xtts", "env_vc", "keys"):
        assert expected in ids
    assert sum(rep["summary"].values()) == len(rep["checks"])
    assert Path(rep["saved_to"]).read_text().startswith("VoiceCloningTTS self-check")
    assert Path(rep["saved_to"]).with_suffix(".json").exists()


async def test_crashing_check_does_not_stop_others(state, monkeypatch):
    def boom(self):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(Doctor, "check_ffmpeg", boom)
    monkeypatch.setattr(Doctor, "check_network", lambda self: [])
    rep = await Doctor(state).run()
    ff = next(c for c in rep["checks"] if c["id"] == "ffmpeg")
    assert ff["status"] == FAIL and "kaboom" in ff["detail"]
    assert any(c["id"] == "keys" for c in rep["checks"])  # later checks still ran


def test_env_probe_reports_imports_and_torch(tmp_path):
    ok = probe_env(Path(sys.executable), "import json", ["pytest"])
    assert ok["import_ok"] is True and "pytest" in ok["packages"]
    bad = probe_env(Path(sys.executable), "import definitely_not_a_module", [])
    assert bad["import_ok"] is False and "ModuleNotFoundError" in bad["import_error"]
    missing = probe_env(tmp_path / "nope" / "python", "pass", [])
    assert "error" in missing


def test_env_check_flags_cpu_only_torch(state, monkeypatch):
    state.gpu_info = GpuInfo(cuda=True, name="NVIDIA GeForce GTX 1650", vram_gb=4.0, cc=(7, 5))
    monkeypatch.setattr(doc, "probe_env", lambda *a, **k: {"import_ok": True, "torch": "2.6.0+cpu",
                                                          "torch_cuda": None, "cuda_available": False})
    c = Doctor(state)._env_check("env_vc", "VC env", Path(sys.executable), "pass", [], None, [], "")
    assert c.status == FAIL and "CPU-only" in c.detail
    monkeypatch.setattr(doc, "probe_env", lambda *a, **k: {"import_ok": True, "torch": "2.6.0+cu124",
                                                          "torch_cuda": "12.4", "cuda_available": True,
                                                          "device": "NVIDIA GeForce GTX 1650"})
    c = Doctor(state)._env_check("env_vc", "VC env", Path(sys.executable), "pass", [], None, [], "")
    assert c.status == OK and "GTX 1650" in c.detail
    assert Doctor(state)._env_check("x", "x", None, "", [], None, [], "fix it").status == WARN


def test_disk_and_audio_warnings(state, monkeypatch):
    import shutil as sh

    monkeypatch.setattr(sh, "disk_usage", lambda p: type("U", (), {"free": 5 * 1024**3})())
    checks = Doctor(state).check_system()
    assert by_id(checks, "disk").status == WARN
    from vctts.audio import devices

    monkeypatch.setattr(devices, "available", lambda: (False, "PortAudio library not found"))
    audio = Doctor(state).check_audio()
    assert audio[0].status == FAIL


def test_models_and_hf_cache(tmp_path, state, monkeypatch):
    repo = tmp_path / "models--ResembleAI--chatterbox-turbo" / "snapshots" / "abc"
    repo.mkdir(parents=True)
    assert hf_cached("ResembleAI/chatterbox-turbo", tmp_path)
    assert not hf_cached("ResembleAI/chatterbox", tmp_path)
    monkeypatch.setattr(doc, "_pkg_version", lambda name: "0.1.7" if name == "chatterbox-tts" else None)
    monkeypatch.setattr(doc, "hf_cache_dir", lambda: tmp_path)
    c = Doctor(state).check_models()
    assert c.status == OK and "1 of 2 downloaded" in c.detail and "Chatterbox English" in c.detail


def test_redaction_and_format():
    assert redact("bad key sk-abcdefghijkl", ["sk-abcdefghijkl"]) == "bad key ***"
    rep = {"version": "0.1.0", "created_at": "now", "deep": False, "duration_s": 1.0,
           "summary": {OK: 1, WARN: 1, FAIL: 0, SKIP: 0},
           "checks": [{"id": "a", "title": "A", "status": OK, "detail": "fine", "fix": ""},
                      {"id": "b", "title": "B", "status": WARN, "detail": "meh", "fix": "do x", "data": {}}]}
    text = format_report(rep)
    assert "[ OK ] A: fine" in text and "[WARN] B: meh" in text and "→ do x" in text


@pytest.fixture
def client(paths, monkeypatch):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    monkeypatch.setattr(Doctor, "check_network", lambda self: [doc.Check("net:huggingface", "net", OK, "fake")])
    s = AppState(paths, use_keyring=False)
    s.secrets.set("anthropic", "sk-ant-supersecretvalue123")
    s.providers.register_override("anthropic", FakeProvider([[TextDelta("OK"), Done("end_turn")]]))
    s.tts.register_engine(FakeEngine())
    with TestClient(create_app(s)) as c:
        c.state_obj = s
        yield c


def test_deep_check_via_api(client, speech_mp3):
    with open(speech_mp3, "rb") as f:
        up = client.post("/api/voices/upload", files={"file": ("me.mp3", f, "audio/mpeg")}).json()
    client.post("/api/voices", json={"upload_id": up["upload_id"], "name": "Me"})
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        rep = client.post("/api/doctor", json={"deep": True}).json()
        seen = set()
        while True:
            e = ws.receive_json()
            if e["type"] == "doctor":
                seen.add(e["check"]["id"])
            if e["type"] == "doctor_done":
                break
    checks = {c["id"]: c for c in rep["checks"]}
    assert checks["deep_tts"]["status"] in (OK, WARN) and "Me" in checks["deep_tts"]["detail"]
    assert checks["deep_vc"]["status"] == OK and "load" in checks["deep_vc"]["detail"]
    assert checks["deep_llm"]["status"] == OK and "OK" in checks["deep_llm"]["detail"]
    assert checks["deep_stt"]["status"] in (SKIP, OK, WARN)
    assert {"deep_tts", "deep_vc", "deep_llm"} <= seen
    assert "supersecretvalue" not in str(rep)
    assert checks["keys"]["status"] == OK
    last = client.get("/api/doctor/last").json()
    assert last["report"]["summary"] == rep["summary"] and last["running"] is False
