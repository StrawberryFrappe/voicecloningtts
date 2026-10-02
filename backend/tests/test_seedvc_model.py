"""Seed-VC streaming converter on the real architecture with random weights.

Opt-in: needs torch + Seed-VC deps (the .venv-vc environment) and
VCTTS_SEEDVC_DIR pointing at a Seed-VC checkout. Skipped otherwise.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")
pytest.importorskip("munch")
SEEDVC = os.environ.get("VCTTS_SEEDVC_DIR")
pytestmark = pytest.mark.skipif(not SEEDVC or not (Path(SEEDVC) / "modules").is_dir(),
                                reason="VCTTS_SEEDVC_DIR not set")

from vctts.vc.seedvc import SeedVCModel, StreamingConverter, StreamSettings  # noqa: E402

from .conftest import make_speechlike  # noqa: E402


@pytest.fixture(scope="module")
def model():
    m = SeedVCModel(Path(SEEDVC), device="cpu")
    cwd = os.getcwd()
    try:
        m.load(random_init=True)
    finally:
        os.chdir(cwd)
    return m


def test_streaming_shapes_gate_and_continuity(model, tmp_path):
    ref = tmp_path / "ref.wav"
    sf.write(ref, make_speechlike(5.0), 24000)
    model.set_reference(str(ref), 3.0)
    conv = StreamingConverter(model, 48000, StreamSettings(diffusion_steps=2, block_time=0.2))
    assert conv.block_frames == 9600 and 380 < conv.latency_ms < 460
    src = np.repeat(make_speechlike(2.0, 24000), 2)
    outs = [conv.process(src[i:i + conv.block]) for i in range(0, src.size - conv.block + 1, conv.block)]
    out = np.concatenate(outs)
    assert out.dtype == np.float32 and np.isfinite(out).all() and np.abs(out).max() <= 1.0
    assert np.abs(out).max() > 0  # inference ran while voiced
    for _ in range(StreamSettings().hangover_blocks + 2):
        silent = conv.process(np.zeros(conv.block, np.float32))
    assert np.abs(silent).max() == 0 and conv.last_infer_ms < 5  # gate skips the model
    with pytest.raises(ValueError):
        conv.process(np.zeros(10, np.float32))


def test_precision_flag_controls_half(model):
    # CPU never uses half; a CUDA fp32 model must not either (GTX 16-series safety).
    assert model.fp16 is False
    m = SeedVCModel(Path(SEEDVC), device="cpu", fp16=False)
    assert m.fp16 is False


FT = os.environ.get("VCTTS_TEST_FT_CKPT")


@pytest.mark.skipif(not FT, reason="VCTTS_TEST_FT_CKPT not set")
def test_finetune_weights_swap_benchmark_and_convert(model, tmp_path):
    from vctts.vc.seedvc import benchmark, convert_file

    ref = tmp_path / "ref.wav"
    sf.write(ref, make_speechlike(5.0), 24000)
    settings = StreamSettings(diffusion_steps=2, block_time=0.3)
    src = np.repeat(make_speechlike(1.5, 24000), 2)

    def run_once():
        import torch

        torch.manual_seed(0)
        model.set_reference(str(ref), 3.0)
        conv = StreamingConverter(model, 48000, settings)
        return np.concatenate([conv.process(src[i:i + conv.block])
                               for i in range(0, src.size - conv.block + 1, conv.block)])

    model.set_dit_weights(None)
    base = run_once()
    model.set_dit_weights(FT)
    assert model.dit_key == FT
    tuned = run_once()
    assert not np.allclose(base, tuned)  # different weights → different output
    model.set_dit_weights(None)
    np.testing.assert_allclose(run_once(), base, atol=1e-5)  # base weights restored exactly

    b = benchmark(model, 48000, settings, blocks=2)
    assert b["infer_ms"] > 0 and b["block_ms"] == 300.0
    inp = tmp_path / "in.wav"
    sf.write(inp, make_speechlike(2.0), 24000)
    secs = convert_file(model, str(inp), str(tmp_path / "out.wav"), settings)
    out, sr = sf.read(tmp_path / "out.wav")
    assert sr == 48000 and abs(secs - 2.0) < 0.05 and np.isfinite(out).all()
