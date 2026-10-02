"""Turn an uploaded mp3/mp4/wav/... into a clean mono reference clip."""

from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

REFERENCE_SR = 24000
MIN_SECONDS = 3.0
RECOMMENDED_MIN = 8.0
RECOMMENDED_MAX = 30.0
MAX_SECONDS = 60.0


class IngestError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise IngestError(
            "ffmpeg not found. Install it or `pip install imageio-ffmpeg`."
        ) from e


def extract_audio(src: Path, dst: Path, sr: int = REFERENCE_SR) -> Path:
    """Decode any audio/video container to mono 16-bit WAV at ``sr``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0 or not dst.exists():
        msg = (proc.stderr or "").strip().splitlines()
        raise IngestError(
            "Could not read audio from this file" + (f": {msg[-1]}" if msg else "")
        )
    return dst


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def waveform_peaks(audio: np.ndarray, buckets: int = 1000) -> list[float]:
    """Max-abs peaks per bucket for drawing a waveform in the UI."""
    if audio.size == 0:
        return []
    buckets = max(1, min(buckets, audio.size))
    usable = audio[: (audio.size // buckets) * buckets]
    peaks = np.abs(usable.reshape(buckets, -1)).max(axis=1)
    top = float(peaks.max()) or 1.0
    return [round(float(p) / top, 4) for p in peaks]


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0, pad_s: float = 0.15) -> np.ndarray:
    if audio.size == 0:
        return audio
    frame = max(1, int(sr * 0.02))
    n = audio.size // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    voiced = np.where(db > threshold_db)[0]
    if voiced.size == 0:
        return audio
    pad = int(pad_s * sr)
    start = max(0, voiced[0] * frame - pad)
    end = min(audio.size, (voiced[-1] + 1) * frame + pad)
    return audio[start:end]


def normalize(audio: np.ndarray, target_rms_db: float = -20.0, peak_limit: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))  # remove DC
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-6:
        return audio
    gain = 10 ** (target_rms_db / 20) / rms
    out = audio * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_limit:
        out *= peak_limit / peak
    return out.astype(np.float32)


def make_reference(
    source_wav: Path,
    dst: Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict:
    """Cut [start, end] from a decoded upload, clean it up and save it."""
    audio, sr = load_mono(source_wav)
    total = audio.size / sr
    s = max(0.0, start_s or 0.0)
    e = min(total, end_s if end_s is not None else total)
    if e - s > MAX_SECONDS:
        e = s + MAX_SECONDS
    seg = audio[int(s * sr): int(e * sr)]
    seg = trim_silence(seg, sr)
    duration = seg.size / sr
    if duration < MIN_SECONDS:
        raise IngestError(
            f"The selected clip has only {duration:.1f}s of speech; select at least {MIN_SECONDS:.0f}s "
            f"(ideally {RECOMMENDED_MIN:.0f}-{RECOMMENDED_MAX:.0f}s)."
        )
    seg = normalize(seg)
    if sr != REFERENCE_SR:
        from ..audio.resample import resample

        seg = resample(seg, sr, REFERENCE_SR)
        sr = REFERENCE_SR
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), seg, sr, subtype="PCM_16")
    warnings = []
    if duration < RECOMMENDED_MIN:
        warnings.append(
            f"Clip is {duration:.1f}s; {RECOMMENDED_MIN:.0f}-{RECOMMENDED_MAX:.0f}s of clean speech clones better."
        )
    return {"duration": round(duration, 2), "sample_rate": sr, "warnings": warnings}
