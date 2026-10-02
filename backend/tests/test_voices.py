import numpy as np
import pytest
import soundfile as sf

from vctts.voices import IngestError, VoiceLibrary, extract_audio, waveform_peaks
from vctts.voices.ingest import REFERENCE_SR, load_mono


@pytest.mark.parametrize("fixture", ["speech_mp3", "speech_mp4", "speech_wav"])
def test_extract_from_mp3_mp4_wav(request, tmp_path, fixture):
    src = request.getfixturevalue(fixture)
    out = extract_audio(src, tmp_path / "decoded.wav")
    audio, sr = load_mono(out)
    assert sr == REFERENCE_SR
    assert 5.5 < audio.size / sr < 6.5
    assert np.max(np.abs(audio)) > 0.05
    peaks = waveform_peaks(audio, 100)
    assert len(peaks) == 100 and max(peaks) == 1.0


def test_extract_rejects_garbage(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(IngestError):
        extract_audio(bad, tmp_path / "o.wav")


def test_library_create_trim_normalize_export_import(paths, speech_mp4, tmp_path):
    lib = VoiceLibrary(paths.voices)
    decoded = extract_audio(speech_mp4, tmp_path / "d.wav")
    v, warnings = lib.create_from_audio(decoded, "Alice", language="es", engine="xtts", start_s=0.0, end_s=6.0)
    ref, sr = sf.read(lib.reference_path(v.id))
    # leading/trailing silence trimmed (~0.5s each side, minus padding)
    assert 4.8 < v.duration < 5.6
    assert abs(20 * np.log10(np.sqrt(np.mean(ref ** 2))) + 20) < 1.5  # ~-20 dBFS RMS
    assert warnings  # < 8s recommended minimum
    assert [x.name for x in lib.list()] == ["Alice"]

    v2 = lib.update(v.id, name="Alicia", engine_settings={"xtts": {"speed": 1.2}})
    assert v2.name == "Alicia" and v2.engine_settings["xtts"]["speed"] == 1.2

    blob = lib.export_zip(v.id)
    imported = lib.import_zip(blob)
    assert imported.id != v.id and imported.name == "Alicia" and imported.language == "es"
    assert len(lib.list()) == 2
    lib.delete(v.id)
    assert [x.id for x in lib.list()] == [imported.id]


def test_library_rejects_too_short_selection(paths, speech_wav):
    lib = VoiceLibrary(paths.voices)
    with pytest.raises(IngestError):
        lib.create_from_audio(speech_wav, "Short", start_s=1.0, end_s=2.0)
    assert lib.list() == []


def test_import_rejects_bad_zip(paths):
    with pytest.raises(IngestError):
        VoiceLibrary(paths.voices).import_zip(b"nope")
