import numpy as np

from vctts.integrations.discord.sink import FRAME_BYTES, SILENCE, DiscordVoiceSink


def test_discord_sink_produces_48k_stereo_20ms_frames():
    sink = DiscordVoiceSink()
    audio = np.full(24000, 0.5, np.float32)  # 1 s at 24 kHz
    sink.play(audio, 24000, "s1")
    assert abs(sink.pending_seconds() - 1.0) < 0.03
    frame = sink.read_frame()
    assert len(frame) == FRAME_BYTES
    samples = np.frombuffer(frame, "<i2").reshape(-1, 2)
    assert np.all(samples[:, 0] == samples[:, 1])
    assert abs(samples[100, 0] - 16383) < 200
    sink.stop()
    assert sink.read_frame() == SILENCE
