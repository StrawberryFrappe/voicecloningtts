"""Stand-in for vctts.vc.train used by tests (same CLI and progress protocol)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def emit(**m):
    print(json.dumps(m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir")
ap.add_argument("--out")
ap.add_argument("--steps", type=int)
ap.add_argument("--batch-size", type=int)
ap.add_argument("--precision")
ap.add_argument("--device")
ap.add_argument("--delay", type=float, default=0.01)
ap.add_argument("--fail", action="store_true")
a = ap.parse_args()

files = sorted(os.listdir(a.dataset_dir))
emit(event="status", message=f"Loading models… ({len(files)} files, {a.precision}, bs={a.batch_size})")
for i in range(1, a.steps + 1):
    time.sleep(a.delay)
    if a.fail and i == 3:
        emit(event="error", error="CUDA out of memory")
        sys.exit(1)
    emit(event="progress", step=i, max_steps=a.steps, loss=round(10 / i, 3), sec_per_step=a.delay)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_bytes(b"fake-weights")
emit(event="done", out=a.out, steps=a.steps)
