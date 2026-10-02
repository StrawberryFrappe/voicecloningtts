"""Sample-rate conversion helpers."""

from __future__ import annotations

from math import gcd

import numpy as np


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality polyphase resampling of a mono or (frames, channels) array."""
    if sr_in == sr_out or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    out = resample_poly(audio, up, down, axis=0)
    return out.astype(np.float32, copy=False)


def to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Convert mono (n,) or (n, c) audio to exactly (n, channels)."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == channels:
        return audio
    mono = audio.mean(axis=1, keepdims=True)
    return np.repeat(mono, channels, axis=1)
