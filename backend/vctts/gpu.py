"""GPU detection and VRAM budgeting.

Target hardware includes 4 GB cards like the GTX 1650, which cannot hold the
TTS model, the voice changer and Whisper at the same time, and whose fp16
path is unreliable (GTX 16-series cards are known to produce NaNs/silence in
half precision). This module decides:

* **memory mode**: ``low`` keeps a single heavy model on the GPU at a time
  (:class:`GpuCoordinator` unloads the others), ``normal`` changes nothing.
* **precision**: ``fp32`` on GTX 16-series / pre-Volta cards, ``fp16`` otherwise.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Callable

log = logging.getLogger(__name__)

LOW_VRAM_GB = 6.0


@dataclass
class GpuInfo:
    cuda: bool = False
    name: str = ""
    vram_gb: float = 0.0
    cc: tuple[int, int] = (0, 0)

    @property
    def gtx16(self) -> bool:
        return "GTX 16" in self.name.upper()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cc"] = f"{self.cc[0]}.{self.cc[1]}" if self.cuda else None
        return d


_cached: GpuInfo | None = None
_lock = threading.Lock()


def detect(refresh: bool = False) -> GpuInfo:
    """Describe GPU 0 using the main environment's torch (absent → no CUDA)."""
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        info = GpuInfo()
        try:
            import torch

            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                info = GpuInfo(cuda=True, name=p.name, vram_gb=round(p.total_memory / 1024**3, 1),
                               cc=(p.major, p.minor))
        except Exception as e:  # torch missing or driver trouble
            log.info("no CUDA GPU detected (%s)", e)
        _cached = info
        return info


def resolve_low_vram(mode: str, info: GpuInfo) -> bool:
    if mode == "low":
        return True
    if mode == "normal":
        return False
    return info.cuda and info.vram_gb < LOW_VRAM_GB


def resolve_precision(pref: str, info: GpuInfo) -> str:
    if pref in ("fp16", "fp32"):
        return pref
    if not info.cuda or info.gtx16 or info.cc < (7, 0):
        return "fp32"
    return "fp16"


class GpuBusy(RuntimeError):
    pass


FEATURE_NAMES = {"tts": "the TTS voice", "vc": "the voice changer", "stt": "speech-to-text",
                 "train": "fine-tuning"}


class GpuCoordinator:
    """Keeps a single heavy GPU feature resident when VRAM is tight.

    Features register a ``release`` callback that frees their VRAM. In low
    mode, ``acquire(x)`` releases every other holder first; in normal mode it
    only records the holder. ``train`` can be made exclusive: while it runs,
    other features are refused instead of evicting it.
    """

    def __init__(self, low_vram: Callable[[], bool]):
        self._low_vram = low_vram
        self._release: dict[str, Callable[[], None]] = {}
        self._holders: set[str] = set()
        self._exclusive: str | None = None
        self._lock = threading.RLock()
        self.listeners: list[Callable[[dict], None]] = []

    @property
    def low_vram(self) -> bool:
        return bool(self._low_vram())

    @property
    def holders(self) -> set[str]:
        return set(self._holders)

    @property
    def exclusive(self) -> str | None:
        return self._exclusive

    def register(self, feature: str, release: Callable[[], None]) -> None:
        self._release[feature] = release

    def acquire(self, feature: str) -> list[str]:
        """Claim the GPU for ``feature``; returns the features that were evicted."""
        with self._lock:
            if self._exclusive and self._exclusive != feature:
                raise GpuBusy(f"{FEATURE_NAMES.get(self._exclusive, self._exclusive).capitalize()} "
                              f"is using the GPU; try again when it finishes.")
            evicted: list[str] = []
            if self.low_vram:
                for other in sorted(self._holders - {feature}):
                    fn = self._release.get(other)
                    try:
                        if fn:
                            fn()
                    except Exception:
                        log.exception("releasing %s failed", other)
                    self._holders.discard(other)
                    evicted.append(other)
            self._holders.add(feature)
        if evicted:
            log.info("low-VRAM: %s evicted %s", feature, evicted)
            for cb in list(self.listeners):
                try:
                    cb({"type": "gpu_swap", "acquired": feature, "evicted": evicted})
                except Exception:
                    pass
        return evicted

    def release(self, feature: str) -> None:
        with self._lock:
            self._holders.discard(feature)
            if self._exclusive == feature:
                self._exclusive = None

    def begin_exclusive(self, feature: str) -> list[str]:
        """Evict everyone (regardless of mode) and block others until release()."""
        with self._lock:
            if self._exclusive and self._exclusive != feature:
                raise GpuBusy(f"{FEATURE_NAMES.get(self._exclusive, self._exclusive).capitalize()} is already running.")
            self._exclusive = feature  # set first so release callbacks can tell why they're evicted
            evicted = []
            for other in sorted(self._holders - {feature}):
                fn = self._release.get(other)
                try:
                    if fn:
                        fn()
                except Exception:
                    log.exception("releasing %s failed", other)
                self._holders.discard(other)
                evicted.append(other)
            self._holders.add(feature)
            return evicted
