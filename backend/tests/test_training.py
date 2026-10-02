"""Fine-tuning data/jobs, low-VRAM swapping, presets/auto-tune (fake workers, no GPU)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vctts.gpu import GpuBusy, GpuCoordinator
from vctts.main import create_app
from vctts.state import AppState
from vctts.vc import StreamSettings
from vctts.vc.changer import PRESETS, autotune, preset_settings
from vctts.vc.training import TrainingManager
from vctts.voices import VoiceLibrary
from vctts.voices.ingest import split_on_silence

from .conftest import FakeEngine, FakeProvider, make_speechlike


def long_recording(sr=24000, utterances=6, utt_s=4.0, gap_s=0.8):
    parts = []
    for _ in range(utterances):
        parts.append(make_speechlike(utt_s + 1.0, sr)[int(0.5 * sr):-int(0.5 * sr)])
        parts.append(np.zeros(int(gap_s * sr), np.float32))
    return np.concatenate(parts)


def test_split_on_silence_cuts_at_pauses():
    sr = 24000
    clips = split_on_silence(long_recording(sr), sr)
    assert len(clips) >= 3
    assert all(3.0 <= c.size / sr <= 12.0 for c in clips)
    # no pauses at all → hard cuts at 12 s
    tone = make_speechlike(30.0, sr)
    tone[:] = 0.2 * np.sin(np.arange(tone.size) * 0.05)
    hard = split_on_silence(tone, sr)
    assert len(hard) >= 2 and max(c.size for c in hard) <= 12 * sr


def make_voice(lib: VoiceLibrary, speech_wav: Path):
    v, _ = lib.create_from_audio(speech_wav, "Me")
    return v


def test_training_clips_and_dataset(paths, speech_wav, tmp_path):
    lib = VoiceLibrary(paths.voices)
    v = make_voice(lib, speech_wav)
    rec = tmp_path / "long.wav"
    sf.write(rec, long_recording(), 24000)
    clips = lib.add_training_audio(v.id, rec)
    assert clips[0]["name"] == "reference.wav" and not clips[0]["removable"]
    assert len(clips) >= 4
    lib.delete_training_clip(v.id, clips[1]["name"])
    assert len(lib.list_training(v.id)) == len(clips) - 1
    with pytest.raises(KeyError):
        lib.delete_training_clip(v.id, "../meta.json")
    ds = lib.build_dataset(v.id)
    assert (ds / "reference.wav").exists() and len(list(ds.glob("*.wav"))) == len(clips) - 1


def test_finetune_zip_roundtrip(paths, speech_wav, tmp_path):
    lib = VoiceLibrary(paths.voices)
    v = make_voice(lib, speech_wav)
    sf.write(tmp_path / "long.wav", long_recording(), 24000)
    lib.add_training_audio(v.id, tmp_path / "long.wav")
    lib.finetune_path(v.id).parent.mkdir(parents=True, exist_ok=True)
    lib.finetune_path(v.id).write_bytes(b"weights")
    lib.set_finetune(v.id, {"steps": 100})
    assert lib.active_finetune(v.id) == lib.finetune_path(v.id)
    lib.update(v.id, vc_use_finetune=False)
    assert lib.active_finetune(v.id) is None
    imported = lib.import_zip(lib.export_zip(v.id))
    assert imported.vc_finetune == {"steps": 100}
    assert lib.finetune_path(imported.id).read_bytes() == b"weights"
    assert len(lib.list_training(imported.id)) == len(lib.list_training(v.id))
    lib.set_finetune(v.id, None)
    assert not lib.finetune_path(v.id).exists() and lib.get(v.id).vc_finetune is None


def make_manager(lib, gpu, events, extra=None, monkeypatch=None):
    return TrainingManager(lib, gpu, on_event=events.append, python=Path(sys.executable),
                           module="tests.fake_trainer", extra_args=extra or [],
                           extra_env={"VCTTS_VC_FAKE": "1"})


def wait_train(mgr, states, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mgr.status()["state"] in states:
            return mgr.status()
        time.sleep(0.02)
    raise AssertionError(mgr.status())


def test_finetune_job_progress_result_and_gpu_exclusivity(paths, speech_wav, monkeypatch):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    lib = VoiceLibrary(paths.voices)
    v = make_voice(lib, speech_wav)
    released = []
    gpu = GpuCoordinator(lambda: False)
    gpu.register("tts", lambda: released.append("tts"))
    gpu.acquire("tts")
    events = []
    mgr = make_manager(lib, gpu, events, ["--delay", "0.02"])
    st = mgr.start(v.id, steps=60)
    assert st["state"] == "starting" and released == ["tts"]  # training evicts even in normal mode
    with pytest.raises(GpuBusy):
        gpu.acquire("vc")
    done = wait_train(mgr, {"done", "error"})
    assert done["state"] == "done", done
    assert lib.get(v.id).vc_finetune["steps"] == 60
    assert lib.finetune_path(v.id).read_bytes() == b"fake-weights"
    progress = [e for e in events if e.get("step")]
    assert progress and progress[-1]["step"] == 60 and progress[-1]["eta_s"] == 0
    gpu.acquire("vc")  # released after training


def test_finetune_failure_and_cancel(paths, speech_wav, monkeypatch):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    lib = VoiceLibrary(paths.voices)
    v = make_voice(lib, speech_wav)
    gpu = GpuCoordinator(lambda: True)
    mgr = make_manager(lib, gpu, [], ["--fail"])
    mgr.start(v.id, steps=60)
    st = wait_train(mgr, {"error", "done"})
    assert st["state"] == "error" and "out of memory" in st["error"]
    assert lib.get(v.id).vc_finetune is None and gpu.exclusive is None

    mgr = make_manager(lib, gpu, [], ["--delay", "0.2"])
    mgr.start(v.id, steps=500)
    wait_train(mgr, {"running"})
    mgr.cancel()
    assert wait_train(mgr, {"cancelled"})["state"] == "cancelled"
    assert lib.get(v.id).vc_finetune is None and gpu.exclusive is None


def test_defaults_follow_vram():
    lib = VoiceLibrary(Path("/tmp/none"))
    assert TrainingManager(lib, GpuCoordinator(lambda: True)).defaults()["batch_size"] == 1
    assert TrainingManager(lib, GpuCoordinator(lambda: False)).defaults()["steps"] == 500


def test_autotune_picks_best_preset_that_keeps_up():
    def measure(cost_per_step):
        def m(st: StreamSettings):
            block_ms = 1000 * st.block_time
            infer = cost_per_step * st.diffusion_steps
            return {"infer_ms": infer, "block_ms": block_ms, "load": infer / block_ms}
        return m

    assert autotune(measure(5.0))["preset"] == "quality"      # fast GPU
    r = autotune(measure(40.0))                                 # GTX 1650-ish
    assert r["preset"] == "low" and r["ok"] and [t["preset"] for t in r["tried"]] == ["quality", "balanced", "low"]
    slow = autotune(measure(500.0))                             # nothing keeps up
    assert slow["preset"] == "minimal" and not slow["ok"]
    assert preset_settings("low").inference_cfg_rate == 0.0 and list(PRESETS)[0] == "quality"


# -- app-level: low-VRAM swapping + API -------------------------------------------

@pytest.fixture
def low_client(paths, monkeypatch):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    state = AppState(paths, use_keyring=False)
    state.db.set_setting("gpu_memory_mode", "low")
    state.providers.register_override("fake", FakeProvider())
    engine = FakeEngine()
    state.tts.register_engine(engine)
    state.training = TrainingManager(state.voices, state.gpu, on_event=state._audio_event,
                                     python=Path(sys.executable), module="tests.fake_trainer",
                                     extra_args=["--delay", "0.01"], extra_env={"VCTTS_VC_FAKE": "1"})
    with TestClient(create_app(state)) as c:
        c.state_obj = state
        yield c


def upload_voice(c, speech_mp3):
    with open(speech_mp3, "rb") as f:
        up = c.post("/api/voices/upload", files={"file": ("me.mp3", f, "audio/mpeg")}).json()
    return c.post("/api/voices", json={"upload_id": up["upload_id"], "name": "Me"}).json()["voice"]


def test_low_vram_tts_evicts_vc_and_resumes(low_client, speech_mp3):
    c, s = low_client, low_client.state_obj
    v = upload_voice(c, speech_mp3)
    assert c.get("/api/gpu").json()["low_vram"] is True
    # Simulate a live voice changer without audio hardware.
    s.audio._running = True
    s.audio.config.input_device = "mic"
    s.start_voice_changer(v["id"])
    assert s.vc.active and s.gpu.holders == {"vc"}
    r = c.post("/api/speak", json={"text": "This reply needs the GPU for a moment, then hands it back."})
    assert r.status_code == 200
    deadline = time.time() + 10
    while time.time() < deadline and not (s.vc.active and "tts" not in s.gpu.holders and s.vc.state != "off"):
        time.sleep(0.05)
    assert s.vc.active, "voice changer should resume after speech"
    assert s.gpu.holders == {"vc"}
    s.vc.stop()
    s.audio._running = False


def test_finetune_api_flow(low_client, speech_mp3, tmp_path):
    c, s = low_client, low_client.state_obj
    v = upload_voice(c, speech_mp3)
    sf.write(tmp_path / "long.wav", long_recording(), 24000)
    with open(tmp_path / "long.wav", "rb") as f:
        up = c.post("/api/voices/upload", files={"file": ("long.wav", f, "audio/wav")}).json()
    tr = c.post(f"/api/voices/{v['id']}/training", json={"upload_id": up["upload_id"]}).json()
    assert len(tr["clips"]) >= 4 and tr["seconds"] > 20
    assert c.get(f"/api/voices/{v['id']}/training").json()["seconds"] == tr["seconds"]
    st = c.post(f"/api/voices/{v['id']}/finetune", json={"steps": 80}).json()
    assert st["state"] == "starting" and st["batch_size"] == 1  # low-VRAM default
    assert c.post("/api/speak", json={"text": "blocked while training"}).status_code in (200, 400)
    deadline = time.time() + 20
    while c.get("/api/vc/train").json()["state"] not in ("done", "error") and time.time() < deadline:
        time.sleep(0.05)
    assert c.get("/api/vc/train").json()["state"] == "done"
    voice = next(x for x in c.get("/api/voices").json()["voices"] if x["id"] == v["id"])
    assert voice["vc_finetune"]["steps"] == 80
    assert c.patch(f"/api/voices/{v['id']}", json={"vc_use_finetune": False}).json()["vc_use_finetune"] is False
    # compare base vs fine-tuned on a recording (fake worker halves the audio)
    rid = "rec123"
    sf.write(s.paths.recordings / f"{rid}.wav", make_speechlike(2.0), 24000)
    cmp_ = c.post(f"/api/voices/{v['id']}/finetune/compare", json={"recording_id": rid}).json()
    assert cmp_["base_wav_b64"] and cmp_["finetuned_wav_b64"]
    assert c.delete(f"/api/voices/{v['id']}/finetune").json()["vc_finetune"] is None


def test_presets_and_autotune_api(low_client, speech_mp3):
    c = low_client
    upload_voice(c, speech_mp3)
    p = c.get("/api/vc/presets").json()
    assert p["recommended"] == "low" and p["presets"]["low"]["latency_ms"] == 620
    r = c.post("/api/vc/autotune", json={"fake_ms_per_step": 40.0}).json()
    assert r["preset"] == "low" and r["status"]["settings"]["diffusion_steps"] == 4
    assert c.post("/api/vc/benchmark", json={}).json()["load"] > 0
