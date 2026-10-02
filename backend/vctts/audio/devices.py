"""Audio device discovery (PortAudio via sounddevice).

On Windows we prefer the WASAPI host API: lowest latency in shared mode and it
lets us open any sample rate with auto-conversion.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass

log = logging.getLogger(__name__)

try:  # PortAudio may be missing on headless Linux
    import sounddevice as sd

    _SD_ERROR: str | None = None
except (OSError, ImportError) as e:  # pragma: no cover - platform dependent
    sd = None
    _SD_ERROR = str(e)

# Playback endpoints of known virtual cables. Apps record from the matching
# capture endpoint (e.g. play into "CABLE Input", apps pick "CABLE Output").
VIRTUAL_PLAYBACK_HINTS = (
    "cable input",          # VB-Audio Virtual Cable
    "cable-a input", "cable-b input",
    "voicemeeter input", "voicemeeter aux input", "voicemeeter vaio",
    "blackhole",            # macOS
    "vctts_virtual_mic",    # PulseAudio/PipeWire null sink (Linux helper)
)
VIRTUAL_CAPTURE_HINTS = ("cable output", "voicemeeter output", "blackhole", "vctts_virtual_mic")


@dataclass
class DeviceInfo:
    id: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False
    is_virtual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def available() -> tuple[bool, str]:
    if sd is None:
        return False, f"Audio I/O unavailable: {_SD_ERROR}"
    return True, ""


def preferred_hostapi_index() -> int | None:
    if sd is None:
        return None
    apis = sd.query_hostapis()
    if sys.platform == "win32":
        for i, api in enumerate(apis):
            if "WASAPI" in api["name"]:
                return i
    try:
        return sd.default.hostapi if sd.default.hostapi >= 0 else None  # type: ignore[attr-defined]
    except Exception:
        return None


def list_devices(all_hostapis: bool = False) -> list[DeviceInfo]:
    if sd is None:
        return []
    try:
        sd._terminate()  # refresh the device list (hot-plugged devices)
        sd._initialize()
    except Exception:
        pass
    apis = sd.query_hostapis()
    preferred = preferred_hostapi_index()
    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        if not all_hostapis and preferred is not None and d["hostapi"] != preferred:
            continue
        api = apis[d["hostapi"]]
        name_l = d["name"].lower()
        out.append(
            DeviceInfo(
                id=i,
                name=d["name"],
                hostapi=api["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
                is_default_input=api.get("default_input_device") == i,
                is_default_output=api.get("default_output_device") == i,
                is_virtual=any(h in name_l for h in VIRTUAL_PLAYBACK_HINTS + VIRTUAL_CAPTURE_HINTS),
            )
        )
    return out


def find_virtual_playback(devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    devices = devices if devices is not None else list_devices()
    for hint in VIRTUAL_PLAYBACK_HINTS:
        for d in devices:
            if d.max_output_channels > 0 and hint in d.name.lower():
                return d
    return None


def find_by_name(name: str | None, kind: str, devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    """Resolve a saved device by name (indices change when devices are plugged in)."""
    if not name:
        return None
    devices = devices if devices is not None else list_devices()
    want = name.lower()
    for d in devices:
        ok = d.max_output_channels > 0 if kind == "output" else d.max_input_channels > 0
        if ok and d.name.lower() == want:
            return d
    for d in devices:
        ok = d.max_output_channels > 0 if kind == "output" else d.max_input_channels > 0
        if ok and want in d.name.lower():
            return d
    return None


def default_device(kind: str, devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    devices = devices if devices is not None else list_devices()
    for d in devices:
        if kind == "input" and d.is_default_input:
            return d
        if kind == "output" and d.is_default_output:
            return d
    return None


def virtual_cable_status() -> dict:
    ok, reason = available()
    if not ok:
        return {"installed": False, "device": None, "reason": reason}
    dev = find_virtual_playback()
    return {
        "installed": dev is not None,
        "device": dev.name if dev else None,
        "capture_name": _capture_name_for(dev.name) if dev else None,
        "reason": "" if dev else "No virtual audio cable found",
    }


def _capture_name_for(playback_name: str) -> str:
    n = playback_name
    for a, b in (("Input", "Output"), ("input", "output")):
        if a in n:
            return n.replace(a, b, 1)
    return n
