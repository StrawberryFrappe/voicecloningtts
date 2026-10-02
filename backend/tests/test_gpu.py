import pytest

from vctts.gpu import GpuBusy, GpuCoordinator, GpuInfo, resolve_low_vram, resolve_precision

GTX1650 = GpuInfo(cuda=True, name="NVIDIA GeForce GTX 1650", vram_gb=4.0, cc=(7, 5))
RTX3060 = GpuInfo(cuda=True, name="NVIDIA GeForce RTX 3060", vram_gb=12.0, cc=(8, 6))
GTX1060 = GpuInfo(cuda=True, name="NVIDIA GeForce GTX 1060 6GB", vram_gb=6.0, cc=(6, 1))
CPU = GpuInfo()


def test_policy_resolution():
    assert resolve_low_vram("auto", GTX1650) and resolve_precision("auto", GTX1650) == "fp32"
    assert not resolve_low_vram("auto", RTX3060) and resolve_precision("auto", RTX3060) == "fp16"
    assert not resolve_low_vram("auto", GTX1060) and resolve_precision("auto", GTX1060) == "fp32"  # pre-Volta
    assert not resolve_low_vram("auto", CPU) and resolve_precision("auto", CPU) == "fp32"
    assert resolve_low_vram("low", RTX3060) and not resolve_low_vram("normal", GTX1650)
    assert resolve_precision("fp16", GTX1650) == "fp16"  # explicit override wins
    assert GTX1650.to_dict()["cc"] == "7.5" and CPU.to_dict()["cc"] is None


def make(low: bool):
    released = []
    c = GpuCoordinator(lambda: low)
    for f in ("tts", "vc", "stt"):
        c.register(f, lambda f=f: released.append(f))
    return c, released


def test_low_mode_keeps_one_holder():
    c, released = make(True)
    assert c.acquire("tts") == []
    assert c.acquire("tts") == []  # re-acquiring is free
    assert c.acquire("vc") == ["tts"] and released == ["tts"]
    assert c.acquire("stt") == ["vc"] and c.holders == {"stt"}


def test_normal_mode_never_evicts():
    c, released = make(False)
    c.acquire("tts")
    c.acquire("vc")
    c.acquire("stt")
    assert released == [] and c.holders == {"tts", "vc", "stt"}


def test_training_is_exclusive_in_any_mode():
    c, released = make(False)
    c.acquire("tts")
    c.acquire("vc")
    assert set(c.begin_exclusive("train")) == {"tts", "vc"}
    with pytest.raises(GpuBusy, match="Fine-tuning"):
        c.acquire("vc")
    c.release("train")
    assert c.acquire("vc") == []


def test_listeners_get_swap_events():
    c, _ = make(True)
    events = []
    c.listeners.append(events.append)
    c.acquire("tts")
    c.acquire("vc")
    assert events == [{"type": "gpu_swap", "acquired": "vc", "evicted": ["tts"]}]


def test_release_callbacks_see_exclusive_owner():
    c = GpuCoordinator(lambda: True)
    seen = []
    c.register("vc", lambda: seen.append(c.exclusive))
    c.acquire("vc")
    c.begin_exclusive("train")
    assert seen == ["train"]
