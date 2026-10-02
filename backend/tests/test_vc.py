"""Real-time voice changer: worker protocol, streaming loop, overload handling, API."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from vctts.vc import StreamSettings, VCError, VoiceChanger
from vctts.vc.changer import VCWorkerClient
from vctts.tts.worker import decode_audio, encode_audio

SR = 48000


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    c = VCWorkerClient(python=Path(sys.executable), extra_env={"VCTTS_VC_FAKE": "1"})
    yield c
    c.close()


def test_settings_are_clamped():
    s = StreamSettings.from_dict({"block_time": 5, "extra_time_right": 0, "extra_time": 3, "extra_time_ce": 1,
                                  "diffusion_steps": 0, "bogus": 1})
    assert s.block_time == 1.0 and s.extra_time_right == 0.02
    assert s.extra_time_ce == 3 and s.diffusion_steps == 1


def test_worker_protocol(fake_client):
    assert fake_client.call("load")["device"] == "cpu"
    cfg = fake_client.call("configure", sample_rate=SR, reference_path="x.wav",
                           settings={"block_time": 0.18})
    assert cfg["block_frames"] == 8640
    block = np.linspace(-0.5, 0.5, 8640, dtype=np.float32)
    out = decode_audio(fake_client.call("process", audio=encode_audio(block))["audio"])
    assert np.allclose(out, block * 0.5)
    with pytest.raises(VCError):
        fake_client.call("process", audio=encode_audio(block[:100]))
    assert fake_client.alive  # errors don't kill the worker


def wait_state(vc, state, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if vc.state == state:
            return
        if vc.state == "error":
            raise AssertionError(vc.error)
        time.sleep(0.02)
    raise AssertionError(f"state {vc.state}, wanted {state}")


def test_voice_changer_streams_converted_audio(fake_client, tmp_path):
    events = []
    vc = VoiceChanger(on_event=events.append, client=fake_client)
    assert not vc.active
    vc.start(tmp_path / "ref.wav", "v1", SR, StreamSettings(block_time=0.1))
    assert vc.active
    assert np.all(vc.read(480) == 0)  # silence while warming up: never leak the real voice
    wait_state(vc, "live")
    t = np.arange(SR) / SR
    src = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = []
    for i in range(0, src.size, 480):  # emulate 10 ms audio callbacks
        vc.feed(src[i:i + 480])
        time.sleep(0.002)
        out.append(vc.read(480))
    time.sleep(0.3)
    out = np.concatenate(out)
    assert np.max(np.abs(out)) == pytest.approx(0.2, abs=0.01)  # fake converter halves the signal
    st = vc.status()
    assert st["state"] == "live" and st["block_ms"] == 100.0 and st["latency_ms"] > 0
    vc.stop()
    assert not vc.active and vc.state == "off"
    assert any(e["state"] == "live" for e in events)


class SlowClient:
    """In-process stand-in whose 'inference' takes 3x the block time."""

    def __init__(self):
        self.block = 0

    def call(self, op, **kw):
        if op == "configure":
            self.block = 4800
            return {"block_frames": self.block, "latency_ms": 200}
        if op == "process":
            time.sleep(0.3)
            return {"audio": kw["audio"], "infer_ms": 300.0}
        return {"device": "cpu"}

    def close(self):
        pass


def test_overload_drops_input_instead_of_growing_latency(monkeypatch, tmp_path):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    vc = VoiceChanger(client=SlowClient())
    vc.start(tmp_path / "r.wav", "v", SR, StreamSettings(block_time=0.1))
    wait_state(vc, "live")
    for _ in range(150):  # 1.5 s of audio, arriving in real time
        vc.feed(np.full(480, 0.1, np.float32))
        time.sleep(0.01)
    st = vc.status()
    vc.stop()
    assert st["dropped_blocks"] > 0
    assert st["load"] > 1.0  # UI shows "GPU too slow for these settings"


def test_unavailable_without_install(monkeypatch):
    monkeypatch.delenv("VCTTS_VC_FAKE", raising=False)
    monkeypatch.setenv("VCTTS_VC_PYTHON", "/nonexistent/python")
    monkeypatch.setattr("vctts.vc.changer.REPO_ROOT", Path("/nonexistent"))
    vc = VoiceChanger()
    assert vc.status()["available"] is False
    with pytest.raises(VCError):
        vc.start(Path("x.wav"), "v", SR)


class SlowLoadClient(SlowClient):
    """Model load takes a while; process is instant."""

    def __init__(self):
        super().__init__()
        self.loads = 0
        self.configures = 0

    def call(self, op, **kw):
        if op == "load":
            self.loads += 1
            time.sleep(0.5)
            return {"device": "cpu"}
        if op == "configure":
            self.configures += 1
        if op == "process":
            return {"audio": kw["audio"], "infer_ms": 1.0}
        return super().call(op, **kw)


def test_restart_while_loading_leaves_one_live_run(monkeypatch, tmp_path):
    monkeypatch.setenv("VCTTS_VC_FAKE", "1")
    client = SlowLoadClient()
    vc = VoiceChanger(client=client)
    vc.start(tmp_path / "r.wav", "v", SR, StreamSettings(block_time=0.1))
    time.sleep(0.1)  # still loading
    vc.start(tmp_path / "r.wav", "v2", SR, StreamSettings(block_time=0.1))  # e.g. a settings change
    wait_state(vc, "live")
    time.sleep(0.8)  # let the superseded run finish its slow load
    threads = [t for t in __import__("threading").enumerate() if t.name == "voice-changer"]
    assert vc.voice_id == "v2" and vc.state == "live"
    assert len(threads) == 1
    vc.stop()
