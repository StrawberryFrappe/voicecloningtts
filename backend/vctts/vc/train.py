"""Per-voice Seed-VC fine-tuning (runs in .venv-vc; launched by vctts.vc.training).

Wraps Seed-VC's own ``train.Trainer`` (GPL-3.0, from vendor/seed-vc) with:

* JSON progress lines on stdout instead of tqdm bars,
* no vocoder (training never uses it: saves VRAM on 4 GB cards),
* fp32 content encoder on GTX 16-series cards (upstream hard-codes fp16),
* one atomic save of the DiT weights at the end (no intermediate checkpoints).

Usage::

    python -m vctts.vc.train --dataset-dir <clips> --out <voice>/vc/ft_model.pth \\
        --steps 200 --batch-size 1 --precision fp32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PRESET_CONFIG = "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"


def emit(**msg) -> None:
    sys.__stdout__.write(json.dumps(msg) + "\n")
    sys.__stdout__.flush()


def build_trainer_class(seed_train, precision: str, random_init: bool):
    import torch

    class VoiceFineTuner(seed_train.Trainer):
        def build_vocoder(self, device, config):  # unused by train_one_step
            self.vocoder_fn = None

        def build_sv_model(self, device, config):
            if not random_init:
                return super().build_sv_model(device, config)
            from modules.campplus.DTDNN import CAMPPlus

            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192).eval().to(device)
            self.sv_fn = self.campplus_model

        def build_converter(self, device, config):
            if not random_init:
                return super().build_converter(device, config)
            from modules.openvoice.api import ToneColorConverter

            self.tone_color_converter = ToneColorConverter(
                "modules/openvoice/checkpoints_v2/converter/config.json", device=device)
            self.se_db = torch.randn(8, 256)

        def build_semantic_fn(self, device, config):
            tok = config["model_params"]["speech_tokenizer"]
            if tok.get("type") != "xlsr" or (precision == "fp16" and not random_init):
                return super().build_semantic_fn(device, config)
            from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

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
            w2v = w2v.to(device).eval()
            half = precision == "fp16" and str(device).startswith("cuda")
            if half:
                w2v = w2v.half()
            self.wav2vec_feature_extractor, self.wav2vec_model = fe, w2v

            def semantic_fn(waves_16k):
                inputs = fe([w.cpu().numpy() for w in waves_16k], return_tensors="pt",
                            return_attention_mask=True, padding=True, sampling_rate=16000).to(device)
                with torch.no_grad():
                    vals = inputs.input_values.half() if half else inputs.input_values
                    return w2v(vals).last_hidden_state.float()

            self.semantic_fn = semantic_fn

        def run(self, out_path: Path) -> None:
            self.ema_loss = 0.0
            t0 = time.time()
            last_emit = 0.0
            for epoch in range(self.n_epochs):
                self.epoch = epoch
                for model in self.model.values():
                    model.train()
                for batch in self.train_dataloader:
                    batch = [b.to(self.device) for b in batch]
                    loss = self.train_one_step(batch)
                    self.ema_loss = loss if self.iters == 0 else self.ema_loss * 0.9 + loss * 0.1
                    self.iters += 1
                    now = time.time()
                    if now - last_emit > 1.0 or self.iters >= self.max_steps:
                        last_emit = now
                        emit(event="progress", step=self.iters, max_steps=self.max_steps,
                             loss=round(float(self.ema_loss), 4), sec_per_step=round((now - t0) / self.iters, 3))
                    if self.iters >= self.max_steps:
                        break
                if self.iters >= self.max_steps:
                    break
            state = {"net": {key: self.model[key].state_dict() for key in self.model}}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(suffix=".pth", dir=out_path.parent)
            os.close(fd)
            torch.save(state, tmp)
            os.replace(tmp, out_path)

    return VoiceFineTuner


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--random-init", action="store_true", help="no downloads (offline checks)")
    args = ap.parse_args(argv)

    os.environ.setdefault("TQDM_DISABLE", "1")
    sys.stdout = sys.stderr  # library prints must not corrupt the progress channel
    root = Path(os.environ.get("VCTTS_SEEDVC_DIR") or Path.cwd())
    sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        import torch

        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        emit(event="status", message="Loading models…", device=device)
        import train as seed_train

        if args.random_init:
            seed_train.load_custom_model_from_hf = lambda *a, **k: ""  # skip pretrained DiT download
        trainer_cls = build_trainer_class(seed_train, args.precision, args.random_init)
        with tempfile.TemporaryDirectory() as run_root:
            # Upstream writes logs/configs to ./runs/<name>; keep that out of the repo.
            cfg_path = Path(run_root) / "config.yml"
            cfg_text = (root / PRESET_CONFIG).read_text()
            cfg_path.write_text(cfg_text.replace('log_dir: "./runs/"', f'log_dir: "{Path(run_root).as_posix()}/"'))
            trainer = trainer_cls(config_path=str(cfg_path), pretrained_ckpt_path=None,
                                  data_dir=args.dataset_dir, run_name="ft", batch_size=args.batch_size,
                                  num_workers=0, steps=args.steps, save_interval=10**9, max_epochs=10**6,
                                  device=device)
            emit(event="status", message="Training…")
            trainer.run(Path(args.out))
        emit(event="done", out=str(args.out), steps=args.steps)
    except Exception as e:  # report, then fail
        import traceback

        traceback.print_exc()
        emit(event="error", error=f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
