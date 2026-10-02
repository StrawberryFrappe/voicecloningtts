"""The out-of-process engine protocol, using a fake engine in a real subprocess."""

import sys
from pathlib import Path

import numpy as np
import pytest

from vctts.tts.base import CancelToken, TTSError, VoiceContext
from vctts.tts.remote import RemoteEngine
from vctts.tts.worker import decode_audio, encode_audio


class FakeRemote(RemoteEngine):
    id = "fake-remote"
    display_name = "Fake remote"
    worker_engine = "tests.conftest:FakeEngine"
    cache_key = "fake"

    @classmethod
    def availability(cls):
        return True, ""

    def languages(self):
        return {"en": "English"}

    def settings_schema(self):
        return []


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    eng = FakeRemote(python=Path(sys.executable))
    yield eng
    eng.unload()


@pytest.fixture
def ctx(tmp_path, speech_wav):
    return VoiceContext("v1", speech_wav, "en", lambda key: tmp_path / key)


def test_audio_codec_roundtrip():
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert np.array_equal(decode_audio(encode_audio(x)), x)


def test_remote_load_prepare_synthesize(remote, ctx):
    remote.load()
    assert remote.loaded and remote.device == "cpu"
    remote.prepare_voice(ctx)
    chunks = list(remote.synthesize_stream("Hello from another process.", ctx, CancelToken()))
    assert len(chunks) == 1
    audio, sr = chunks[0]
    assert sr == 24000 and audio.dtype == np.float32 and audio.size > 1000


def test_remote_errors_propagate(remote, tmp_path):
    bad = VoiceContext("v2", tmp_path / "missing.wav", "en", lambda key: tmp_path / key)
    with pytest.raises(TTSError):
        remote.prepare_voice(bad)
    # the worker survives errors
    remote.load()
    assert remote.loaded


def test_remote_cancel_before_start(remote, ctx):
    tok = CancelToken()
    tok.cancel()
    from vctts.tts.base import Cancelled

    with pytest.raises(Cancelled):
        list(remote.synthesize_stream("never spoken", ctx, tok))
    # still usable afterwards
    assert list(remote.synthesize_stream("again", ctx, CancelToken()))


def test_remote_unload_restarts_on_demand(remote, ctx):
    remote.load()
    remote.unload()
    assert not remote.loaded
    assert list(remote.synthesize_stream("back again", ctx, CancelToken()))
