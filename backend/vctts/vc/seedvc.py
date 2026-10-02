"""Seed-VC zero-shot real-time voice conversion (runs inside the .venv-vc worker).

Model: Seed-VC ``seed-uvit-tat-xlsr-tiny`` (https://github.com/Plachtaa/seed-vc),
which converts any input speech to the timbre of a short reference clip with
no training. The streaming scheme (rolling context window, right look-ahead,
SOLA crossfade) follows Seed-VC's real-time GUI, which in turn adapts RVC's.
Seed-VC is GPL-3.0; it is installed separately into ``vendor/seed-vc`` and is
only imported in this worker process.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("vctts.vc")

DIT_REPO = "Plachta/Seed-VC"
DIT_CKPT = "DiT_uvit_tat_xlsr_ema.pth"
DIT_CONFIG = "config_dit_mel_seed_uvit_xlsr_tiny.yml"


@dataclass
class StreamSettings:
    diffusion_steps: int = 8
    inference_cfg_rate: float = 0.7
    max_prompt_length: float = 3.0      # seconds of the reference clip used as prompt
    block_time: float = 0.18            # seconds per processed chunk (latency ~ 2x this)
    crossfade_time: float = 0.04
    extra_time_ce: float = 2.5          # left context for the content encoder
    extra_time: float = 0.5             # left context for the diffusion model
    extra_time_right: float = 0.02      # look-ahead (adds latency)
    gate_db: float = -50.0              # below this the input counts as silence
    hangover_blocks: int = 2            # keep converting this many blocks after speech stops

    @classmethod
    def from_dict(cls, d: dict | None) -> "StreamSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, type(getattr(s, k))(v))
        s.extra_time_ce = max(s.extra_time_ce, s.extra_time)
        s.extra_time_right = max(0.02, s.extra_time_right)
        s.block_time = min(max(s.block_time, 0.06), 1.0)
        s.diffusion_steps = min(max(s.diffusion_steps, 1), 50)
        return s

    def to_dict(self) -> dict:
        return asdict(self)


class SeedVCModel:
    """Loads Seed-VC's real-time model set and runs chunk inference."""

    def __init__(self, root: Path, device: str | None = None, fp16: bool | None = None):
        import torch

        self.root = Path(root)
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.fp16 = (self.device.type == "cuda") if fp16 is None else fp16
        self.ready = False
        self.prompt = None  # (prompt_condition, mel2, style2)
        self.prompt_key: tuple | None = None
        self.dit_key: str | None = None  # fine-tuned checkpoint currently applied (None = base)
        self._base_state: dict | None = None

    def _import_path(self) -> None:
        if not (self.root / "modules").is_dir():
            raise RuntimeError(f"Seed-VC not found at {self.root}")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        os.chdir(self.root)  # seed-vc resolves configs/ and checkpoints/ relative to cwd

    def load(self, random_init: bool = False) -> None:
        """Build the model. ``random_init`` skips downloads (tests / offline checks)."""
        import torch
        import yaml

        self._import_path()
        from modules.campplus.DTDNN import CAMPPlus
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        from modules.hifigan.generator import HiFTGenerator

        dev = self.device
        if random_init:
            cfg_path = self.root / "configs" / "presets" / DIT_CONFIG
            dit_ckpt = None
        else:
            from hf_utils import load_custom_model_from_hf

            dit_ckpt, cfg_path = load_custom_model_from_hf(DIT_REPO, DIT_CKPT, DIT_CONFIG)
        config = yaml.safe_load(open(cfg_path, "r"))
        mp = recursive_munch(config["model_params"])
        mp.dit_type = "DiT"
        model = build_model(mp, stage="DiT")
        if dit_ckpt:
            model, *_ = load_checkpoint(model, None, dit_ckpt, load_only_params=True,
                                        ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(dev)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        if not random_init:
            from hf_utils import load_custom_model_from_hf

            path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
            campplus.load_state_dict(torch.load(path, map_location="cpu"))
        campplus.eval().to(dev)

        hift_cfg = yaml.safe_load(open(self.root / "configs" / "hifigan.yml", "r"))
        vocoder = HiFTGenerator(**hift_cfg["hift"], f0_predictor=ConvRNNF0Predictor(**hift_cfg["f0_predictor"]))
        if not random_init:
            from hf_utils import load_custom_model_from_hf

            path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
            vocoder.load_state_dict(torch.load(path, map_location="cpu"))
        vocoder.eval().to(dev)

        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

        tok = config["model_params"]["speech_tokenizer"]
        if random_init:
            fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                          do_normalize=True, return_attention_mask=True)
            w2v = Wav2Vec2Model(Wav2Vec2Config(hidden_size=1024, num_hidden_layers=tok["output_layer"],
                                               num_attention_heads=16, intermediate_size=4096,
                                               feat_extract_norm="layer", do_stable_layer_norm=True,
                                               conv_bias=True))
        else:
            fe = Wav2Vec2FeatureExtractor.from_pretrained(tok["name"])
            w2v = Wav2Vec2Model.from_pretrained(tok["name"])
        w2v.encoder.layers = w2v.encoder.layers[: tok["output_layer"]]
        w2v = w2v.to(dev).eval()
        # GTX 16-series cards misbehave in half precision; honour fp32 there.
        half = self.fp16 and dev.type == "cuda"
        if half:
            w2v = w2v.half()

        def semantic_fn(waves_16k):
            inputs = fe([w.cpu().numpy() for w in waves_16k], return_tensors="pt",
                        return_attention_mask=True, padding=True, sampling_rate=16000).to(dev)
            with torch.no_grad():
                values = inputs.input_values.half() if half else inputs.input_values
                out = w2v(values)
            return out.last_hidden_state.float()

        sp = config["preprocess_params"]["spect_params"]
        self.sr = int(config["preprocess_params"]["sr"])
        self.hop = int(sp["hop_length"])
        mel_args = {
            "n_fft": sp["n_fft"], "win_size": sp["win_length"], "hop_size": sp["hop_length"],
            "num_mels": sp["n_mels"], "sampling_rate": self.sr, "fmin": sp.get("fmin", 0),
            "fmax": None if sp.get("fmax", "None") == "None" else 8000, "center": False,
        }
        from modules.audio import mel_spectrogram

        self.to_mel = lambda x: mel_spectrogram(x, **mel_args)
        self.model, self.semantic_fn, self.vocoder, self.campplus = model, semantic_fn, vocoder, campplus
        self.prompt = None
        self.prompt_key = None
        self.ready = True
        log.info("Seed-VC loaded on %s (fp16=%s, random_init=%s)", dev, self.fp16, random_init)

    def set_dit_weights(self, path: str | None) -> None:
        """Switch the diffusion model between the base weights and a per-voice fine-tune.

        Only the DiT/length-regulator weights differ (~100 MB), so this is fast;
        the base weights are kept on the CPU to switch back.
        """
        import torch

        key = str(path) if path else None
        if key == self.dit_key:
            return
        if self._base_state is None:
            self._base_state = {k: {n: t.detach().cpu().clone() for n, t in self.model[k].state_dict().items()}
                                for k in self.model}
        if key is None:
            state = self._base_state
        else:
            params = torch.load(key, map_location="cpu")["net"]
            state = {}
            for k in self.model:
                if k not in params:
                    continue
                current = self.model[k].state_dict()
                cleaned = {n[len("module."):] if n.startswith("module.") else n: v for n, v in params[k].items()}
                state[k] = {n: v for n, v in cleaned.items() if n in current and v.shape == current[n].shape}
        for k, sd in state.items():
            self.model[k].load_state_dict(sd, strict=False)
        self.dit_key = key
        self.prompt = None  # prompt condition depends on the length regulator
        log.info("Seed-VC weights: %s", key or "base")

    def set_reference(self, wav_path: str, max_prompt_length: float) -> None:
        import librosa
        import torch
        import torchaudio

        key = (str(wav_path), float(max_prompt_length), os.path.getmtime(wav_path), self.dit_key)
        if self.prompt is not None and self.prompt_key == key:
            return
        ref, _ = librosa.load(wav_path, sr=self.sr)
        ref = ref[: int(self.sr * max_prompt_length)]
        with torch.no_grad():
            ref_t = torch.from_numpy(ref).to(self.device)
            ref_16k = torchaudio.functional.resample(ref_t, self.sr, 16000)
            s_ori = self.semantic_fn(ref_16k.unsqueeze(0))
            feat = torchaudio.compliance.kaldi.fbank(ref_16k.unsqueeze(0), num_mel_bins=80, dither=0,
                                                     sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus(feat.unsqueeze(0))
            mel2 = self.to_mel(ref_t.unsqueeze(0))
            lengths = torch.LongTensor([mel2.size(2)]).to(self.device)
            prompt_condition = self.model.length_regulator(s_ori, ylens=lengths, n_quantizers=3, f0=None)[0]
        self.prompt = (prompt_condition, mel2, style)
        self.prompt_key = key

    def infer_chunk(self, wave_16k, skip_head: int, skip_tail: int, return_length: int,
                    steps: int, cfg_rate: float, ce_dit_difference: float):
        """Convert the rolling 16 kHz window; returns the new region at the model rate."""
        import torch

        if self.prompt is None:
            raise RuntimeError("No reference voice set")
        prompt_condition, mel2, style = self.prompt
        with torch.no_grad():
            s_alt = self.semantic_fn(wave_16k.unsqueeze(0))
            diff = int(ce_dit_difference * 50)
            s_alt = s_alt[:, diff:]
            target = torch.LongTensor(
                [int((skip_head + return_length + skip_tail - diff) / 50 * self.sr // self.hop)]
            ).to(self.device)
            cond = self.model.length_regulator(s_alt, ylens=target, n_quantizers=3, f0=None)[0]
            cat = torch.cat([prompt_condition, cond], dim=1)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32,
                                enabled=self.fp16):
                mel = self.model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(self.device), mel2, style,
                                               None, n_timesteps=steps, inference_cfg_rate=cfg_rate)
                mel = mel[:, :, mel2.size(-1):]
                wave = self.vocoder(mel.float()).squeeze()
        out_len = return_length * self.sr // 50
        tail = skip_tail * self.sr // 50
        return wave[-out_len - tail: -tail].float()


class StreamingConverter:
    """Block-in/block-out converter at the audio device rate (e.g. 48 kHz)."""

    def __init__(self, model: SeedVCModel, sample_rate: int, settings: StreamSettings):
        import torch
        import torchaudio.transforms as tat

        self.m = model
        self.sr = int(sample_rate)
        self.s = settings
        dev = model.device
        zc = self.sr // 50
        rnd = lambda t: int(np.round(t * self.sr / zc)) * zc  # noqa: E731
        self.zc = zc
        self.block = rnd(settings.block_time)
        self.block_16k = 320 * self.block // zc
        self.crossfade = rnd(settings.crossfade_time)
        self.sola_buffer_frame = min(self.crossfade, 4 * zc)
        self.sola_search = zc
        self.extra = rnd(settings.extra_time_ce)
        self.extra_right = max(zc, rnd(settings.extra_time_right))
        total = self.extra + self.crossfade + self.sola_search + self.block + self.extra_right
        self.input = torch.zeros(total, device=dev)
        self.input_16k = torch.zeros(320 * total // zc, device=dev)
        self.sola_buf = torch.zeros(self.sola_buffer_frame, device=dev)
        self.skip_head = self.extra // zc
        self.skip_tail = self.extra_right // zc
        self.return_length = (self.block + self.sola_buffer_frame + self.sola_search) // zc
        fade = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, self.sola_buffer_frame, device=dev)) ** 2
        self.fade_in, self.fade_out = fade, 1 - fade
        self.to_16k = tat.Resample(self.sr, 16000).to(dev)
        self.from_model = tat.Resample(model.sr, self.sr).to(dev) if model.sr != self.sr else None
        self.voiced_left = 0
        self.last_infer_ms = 0.0

    @property
    def block_frames(self) -> int:
        return self.block

    @property
    def latency_ms(self) -> float:
        return 1000 * (2 * self.block + self.extra_right) / self.sr

    def process(self, block: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if block.size != self.block:
            raise ValueError(f"expected {self.block} frames, got {block.size}")
        dev = self.m.device
        x = torch.from_numpy(np.asarray(block, dtype=np.float32)).to(dev)
        self.input[: -self.block] = self.input[self.block:].clone()
        self.input[-self.block:] = x
        self.input_16k[: -self.block_16k] = self.input_16k[self.block_16k:].clone()
        seg = self.to_16k(self.input[-self.block - 2 * self.zc:])[320:]
        self.input_16k[-seg.numel():] = seg

        rms = float(np.sqrt(np.mean(np.square(block)) + 1e-12))
        voiced = 20 * np.log10(rms + 1e-12) > self.s.gate_db
        if voiced:
            self.voiced_left = self.s.hangover_blocks + 1
        out_len = self.input.numel() - self.extra
        t0 = time.perf_counter()
        if self.voiced_left > 0:
            self.voiced_left -= 1
            wav = self.m.infer_chunk(self.input_16k, self.skip_head, self.skip_tail, self.return_length,
                                     self.s.diffusion_steps, self.s.inference_cfg_rate,
                                     self.s.extra_time_ce - self.s.extra_time)
            if self.from_model is not None:
                wav = self.from_model(wav)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        else:
            wav = torch.zeros(out_len, device=dev)
        self.last_infer_ms = 1000 * (time.perf_counter() - t0)
        need = self.block + self.sola_buffer_frame + self.sola_search
        if wav.numel() < need:
            wav = F.pad(wav, (0, need - wav.numel()))

        # SOLA: align the new chunk with the previous tail, then crossfade.
        conv_in = wav[None, None, : self.sola_buffer_frame + self.sola_search]
        nom = F.conv1d(conv_in, self.sola_buf[None, None, :])
        den = torch.sqrt(F.conv1d(conv_in ** 2, torch.ones(1, 1, self.sola_buffer_frame, device=dev)) + 1e-8)
        corr = nom[0, 0] / den[0, 0]
        offset = int(torch.argmax(corr).item()) if corr.numel() > 1 else 0
        wav = wav[offset:].clone()
        wav[: self.sola_buffer_frame] *= self.fade_in
        wav[: self.sola_buffer_frame] += self.sola_buf * self.fade_out
        self.sola_buf[:] = wav[self.block: self.block + self.sola_buffer_frame]
        return wav[: self.block].clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)


def synthetic_voice(seconds: float, sr: int) -> np.ndarray:
    """Continuously voiced test signal (benchmarks must never hit the silence gate)."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 130 + 25 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 10))
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    return (0.25 * sig * env).astype(np.float32)


def benchmark(model: SeedVCModel, sample_rate: int, settings: StreamSettings, blocks: int = 6) -> dict:
    """Median per-block inference time for these settings on this GPU (reference must be set)."""
    conv = StreamingConverter(model, sample_rate, settings)
    src = synthetic_voice((blocks + 1) * conv.block / sample_rate + 0.1, sample_rate)
    times = []
    for i in range(blocks + 1):
        conv.process(src[i * conv.block:(i + 1) * conv.block])
        if i > 0:  # first block includes warm-up
            times.append(conv.last_infer_ms)
    block_ms = 1000 * conv.block / sample_rate
    infer = float(np.median(times))
    return {"infer_ms": round(infer, 1), "block_ms": round(block_ms, 1), "load": round(infer / block_ms, 3),
            "latency_ms": round(conv.latency_ms, 1)}


def convert_file(model: SeedVCModel, in_path: str, out_path: str, settings: StreamSettings,
                 sample_rate: int = 48000) -> float:
    """Offline conversion through the same streaming path the live changer uses."""
    import soundfile as sf

    from ..audio.resample import resample

    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)
    audio = resample(audio.mean(axis=1), sr, sample_rate)
    conv = StreamingConverter(model, sample_rate, settings)
    lag = 2 * conv.block + conv.extra_right  # output trails input by about this much
    padded = np.concatenate([audio, np.zeros(lag + conv.block, np.float32)])
    n = (padded.size // conv.block) * conv.block
    out = np.concatenate([conv.process(padded[i:i + conv.block]) for i in range(0, n, conv.block)])
    out = out[conv.block:conv.block + audio.size]  # drop the start-up delay
    sf.write(out_path, out, sample_rate)
    return out.size / sample_rate
