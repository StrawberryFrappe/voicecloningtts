import numpy as np

from vctts.audio.mixer import MixCore, MixerSettings, RingBuffer, TtsQueue, db_to_gain, render_offline, soft_clip
from vctts.audio.resample import resample, to_channels


def test_ring_buffer_fifo_and_wraparound():
    rb = RingBuffer(10)
    rb.write(np.arange(6, dtype=np.float32))
    assert np.array_equal(rb.read(4), [0, 1, 2, 3])
    rb.write(np.arange(6, 12, dtype=np.float32))  # wraps
    assert rb.available == 8
    assert np.array_equal(rb.read(8), [4, 5, 6, 7, 8, 9, 10, 11])
    assert np.array_equal(rb.read(3), [0, 0, 0])  # underflow -> silence


def test_ring_buffer_overflow_drops_oldest_and_trim():
    rb = RingBuffer(5)
    rb.write(np.arange(8, dtype=np.float32))
    assert np.array_equal(rb.read(5), [3, 4, 5, 6, 7])
    rb.write(np.arange(5, dtype=np.float32))
    assert rb.trim_to(2) == 3
    assert np.array_equal(rb.read(2), [3, 4])


def test_tts_queue_segments_complete_in_order():
    q = TtsQueue()
    q.enqueue("a", np.ones(5, np.float32))
    q.enqueue("b", np.ones(3, np.float32) * 2)
    out, active, done = q.read(6)
    assert active and done == ["a"]
    assert np.array_equal(out, [1, 1, 1, 1, 1, 2])
    out, active, done = q.read(6)
    assert done == ["b"] and np.array_equal(out, [2, 2, 0, 0, 0, 0])
    assert not q.active


def test_mixer_ducks_mic_while_speaking():
    sr = 48000
    core = MixCore(sr, MixerSettings(duck_db=-20, duck_release_ms=0))
    mic = np.full(sr, 0.5, np.float32)
    tts = np.full(sr // 2, 0.1, np.float32)
    core.tts.enqueue("s1", tts)
    out, mon = render_offline(core, mic, block=480)
    # During TTS (after the first ramp block): mic attenuated by 20 dB + tts
    mid = out[sr // 4]
    assert abs(mid - (0.5 * db_to_gain(-20) + 0.1)) < 1e-3
    # After TTS: mic back to full level
    assert abs(out[int(sr * 0.9)] - 0.5) < 1e-3
    # Monitor only carries TTS by default
    assert abs(mon[sr // 4] - 0.1) < 1e-3 and abs(mon[int(sr * 0.9)]) < 1e-6


def test_mixer_mic_disabled_and_mute():
    core = MixCore(16000, MixerSettings(mic_enabled=False))
    out, _ = render_offline(core, np.full(1600, 0.5, np.float32), block=160)
    assert np.allclose(out, 0)
    core = MixCore(16000, MixerSettings(duck_db=-80, duck_release_ms=0))
    core.tts.enqueue("x", np.zeros(800, np.float32))
    out, _ = render_offline(core, np.full(1600, 0.5, np.float32), block=160)
    assert np.allclose(out[200:700], 0)  # muted while "speaking"


def test_soft_clip_is_transparent_and_bounded():
    x = np.array([0.1, -0.5, 0.89, 3.0, -10.0], np.float32)
    y = soft_clip(x)
    assert np.allclose(y[:3], x[:3])
    assert np.all(np.abs(y) <= 1.0)


def test_resample_and_channels():
    x = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
    y = resample(x, 24000, 48000)
    assert y.size == 48000 and y.dtype == np.float32
    assert to_channels(y[:10], 2).shape == (10, 2)


def test_sanitize_bad_driver_input():
    from vctts.audio.mixer import sanitize

    x = np.array([np.nan, np.inf, -np.inf, 1e30, -0.5], np.float32)
    assert np.array_equal(sanitize(x), np.array([0, 1, -1, 1, -0.5], np.float32))
