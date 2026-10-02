"""Self-check ("Diagnose"): verifies every piece of the install and writes one report.

Quick checks take seconds and never download anything. Deep checks actually
load the models (downloading them the first time), speak a sentence, run the
voice changer benchmark, transcribe, and ping the LLM, so the report contains
real timings for this machine.

Every check is independent: a failure is recorded and the run continues.
API keys are never read into the report (and are scrubbed if an error message
happens to echo one).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from . import __version__

if TYPE_CHECKING:  # pragma: no cover
    from .state import AppState

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
ENV_TIMEOUT_S = 180  # first torch import on Windows can be slow
MIN_FREE_GB = 15.0

PROVIDER_HOSTS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai",
}

# (label, cache entry, approx download size in GB)
CHATTERBOX_MODELS = [("Chatterbox Turbo", "ResembleAI/chatterbox-turbo", 1.5),
                     ("Chatterbox English/multilingual", "ResembleAI/chatterbox", 3.0)]
WHISPER_GPU = ("Whisper large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1.6)
WHISPER_CPU = ("Whisper small", "Systran/faster-whisper-small", 0.5)
VC_ENCODER = ("Voice changer encoder (wav2vec2)", "facebook/wav2vec2-xls-r-300m", 1.3)


@dataclass
class Check:
    id: str
    title: str
    status: str
    detail: str = ""
    fix: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cached(repo_id: str, cache: Path | None = None) -> bool:
    d = (cache or hf_cache_dir()) / ("models--" + repo_id.replace("/", "--"))
    snaps = d / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


def xtts_model_dir() -> Path:
    # coqui-tts stores models in the user data dir under "tts"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"


ENV_PROBE = r"""
import json, os, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if out["cuda_available"]:
        out["device"] = torch.cuda.get_device_name(0)
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    exec(os.environ["VCTTS_PROBE_IMPORT"])
    out["import_ok"] = True
except Exception as e:
    out["import_ok"] = False
    out["import_error"] = f"{type(e).__name__}: {e}"
for pkg in os.environ.get("VCTTS_PROBE_PKGS", "").split(","):
    if pkg:
        try:
            from importlib.metadata import version
            out.setdefault("packages", {})[pkg] = version(pkg)
        except Exception:
            pass
print("VCTTS_PROBE " + json.dumps(out))
"""


def probe_env(python: Path, import_code: str, packages: list[str], cwd: Path | None = None,
              extra_path: list[Path] | None = None, timeout: float = ENV_TIMEOUT_S) -> dict:
    """Run a tiny script in another environment and return what it found."""
    env = dict(os.environ)
    env["VCTTS_PROBE_IMPORT"] = import_code
    env["VCTTS_PROBE_PKGS"] = ",".join(packages)
    paths = [str(p) for p in (extra_path or [])]
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env.setdefault("TQDM_DISABLE", "1")
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        proc = subprocess.run([str(python), "-c", ENV_PROBE], capture_output=True, text=True, timeout=timeout,
                              env=env, cwd=str(cwd) if cwd and cwd.is_dir() else None, **kwargs)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout:.0f}s"}
    except OSError as e:
        return {"error": str(e)}
    for line in proc.stdout.splitlines():
        if line.startswith("VCTTS_PROBE "):
            return json.loads(line[len("VCTTS_PROBE "):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": " | ".join(tail) or f"exit code {proc.returncode}"}


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "***")
    return text


# ---------------------------------------------------------------------------
# the doctor


class Doctor:
    def __init__(self, state: "AppState", on_result: Callable[[dict], None] | None = None):
        self.s = state
        self.on_result = on_result or (lambda c: None)
        self.checks: list[Check] = []
        self._tts_audio: tuple[np.ndarray, int] | None = None

    # -- running ---------------------------------------------------------------
    def _record(self, check: Check) -> None:
        self.checks.append(check)
        try:
            self.on_result(check.to_dict())
        except Exception:
            pass

    async def _run(self, check_id: str, title: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:  # a broken check must not stop the others
            result = Check(check_id, title, FAIL, f"Check crashed: {type(e).__name__}: {e}")
        items = result if isinstance(result, list) else [result]
        for c in items:
            c.seconds = round(time.perf_counter() - t0, 2)
            self._record(c)

    async def run(self, deep: bool = False) -> dict:
        started = time.time()
        quick = [
            ("system", "System", self.check_system),
            ("ffmpeg", "ffmpeg", self.check_ffmpeg),
            ("gpu", "GPU (main environment)", self.check_gpu),
            ("env_xtts", "XTTS environment (.venv-xtts)", self.check_env_xtts),
            ("env_vc", "Voice changer environment (.venv-vc)", self.check_env_vc),
            ("audio", "Audio devices", self.check_audio),
            ("engines", "Voice engines installed", self.check_engines),
            ("models", "Model downloads", self.check_models),
            ("network", "Network", self.check_network),
            ("keys", "API keys", self.check_keys),
        ]
        for cid, title, fn in quick:
            await self._run(cid, title, lambda fn=fn: asyncio.to_thread(fn))
        if deep:
            await self._run("deep_tts", "Deep: speak with the cloned voice", self.deep_tts)
            await self._run("deep_vc", "Deep: voice changer speed", self.deep_vc)
            await self._run("deep_stt", "Deep: speech-to-text", self.deep_stt)
            await self._run("deep_llm", "Deep: LLM reply", self.deep_llm)
        return self.report(started, deep)

    # -- report ----------------------------------------------------------------
    def report(self, started: float, deep: bool) -> dict:
        counts = {k: sum(1 for c in self.checks if c.status == k) for k in (OK, WARN, FAIL, SKIP)}
        rep = {
            "version": __version__,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "deep": deep,
            "duration_s": round(time.time() - started, 1),
            "summary": counts,
            "checks": [c.to_dict() for c in self.checks],
        }
        rep["text"] = format_report(rep)
        secrets = [self.s.secrets.get(p) or "" for p in ("openai", "anthropic", "gemini", "openrouter", "discord")]
        rep = json.loads(redact(json.dumps(rep), secrets))
        rep["saved_to"] = self._save(rep)
        return rep

    def _save(self, rep: dict) -> str | None:
        try:
            logs = self.s.paths.logs
            logs.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            txt = logs / f"doctor-{stamp}.txt"
            txt.write_text(rep["text"], "utf-8")
            (logs / f"doctor-{stamp}.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "text"},
                                                                  indent=2), "utf-8")
            for old in sorted(logs.glob("doctor-*.txt"))[:-10]:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
            return str(txt)
        except OSError:
            return None

    # -- quick checks ----------------------------------------------------------
    def check_system(self) -> list[Check]:
        out = []
        info = {"os": platform.platform(), "python": sys.version.split()[0], "app": __version__,
                "data_dir": str(self.s.paths.root)}
        out.append(Check("system", "System", OK, f"{info['os']} · Python {info['python']} · app {__version__}",
                         data=info))
        try:
            probe = self.s.paths.root / ".write-test"
            probe.write_text("x")
            probe.unlink()
            free_gb = shutil.disk_usage(self.s.paths.root).free / 1024**3
            if free_gb < MIN_FREE_GB:
                out.append(Check("disk", "Disk space", WARN, f"{free_gb:.1f} GB free",
                                 "Models still to download need several GB; free up space on this drive.",
                                 {"free_gb": round(free_gb, 1)}))
            else:
                out.append(Check("disk", "Disk space", OK, f"{free_gb:.0f} GB free", data={"free_gb": round(free_gb, 1)}))
        except OSError as e:
            out.append(Check("disk", "Data folder", FAIL, f"Can't write to {self.s.paths.root}: {e}",
                             "Check folder permissions or antivirus blocking %APPDATA%."))
        from .config import frontend_dist_dir

        if frontend_dist_dir():
            out.append(Check("ui", "User interface", OK, "built UI found"))
        else:
            out.append(Check("ui", "User interface", FAIL, "UI not built",
                             "Run `npm install` and `npm run build` in frontend/ (the setup script does this)."))
        return out

    def check_ffmpeg(self) -> Check:
        from .voices.ingest import IngestError, ffmpeg_exe

        try:
            exe = ffmpeg_exe()
            r = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20)
            first = (r.stdout.splitlines() or ["?"])[0]
            return Check("ffmpeg", "ffmpeg", OK if r.returncode == 0 else FAIL, first, data={"path": exe})
        except (IngestError, OSError, subprocess.TimeoutExpired) as e:
            return Check("ffmpeg", "ffmpeg", FAIL, str(e), "pip install imageio-ffmpeg (part of the app install).")

    def check_gpu(self) -> list[Check]:
        if importlib.util.find_spec("torch") is None:
            return [Check("gpu", "GPU (main environment)", FAIL, "PyTorch is not installed in the main environment",
                          "Re-run scripts\\setup_windows.ps1.")]
        import torch

        data = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "low_vram": self.s.low_vram, "precision": self.s.precision}
        if not torch.cuda.is_available():
            if torch.version.cuda is None:
                return [Check("gpu", "GPU (main environment)", FAIL,
                              f"CPU-only PyTorch {torch.__version__} is installed",
                              "Reinstall the CUDA build: re-run scripts\\setup_windows.ps1 (without -Cpu).", data)]
            return [Check("gpu", "GPU (main environment)", WARN,
                          f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) can't see a GPU",
                          "Update the NVIDIA driver (and reboot). Everything will run on the CPU until then.", data)]
        props = torch.cuda.get_device_properties(0)
        data.update(name=props.name, vram_gb=round(props.total_memory / 1024**3, 1), cc=f"{props.major}.{props.minor}")
        out = []
        try:
            a = torch.randn(256, 256, device="cuda")
            ok32 = bool(torch.isfinite(a @ a).all().item())
            h = a.half()
            ok16 = bool(torch.isfinite(h @ h).all().item())
            torch.cuda.synchronize()
        except Exception as e:
            return [Check("gpu", "GPU (main environment)", FAIL, f"{props.name}: CUDA kernels fail: {e}",
                          "Update the NVIDIA driver; make sure no other app holds the GPU in exclusive mode.", data)]
        data.update(fp32_ok=ok32, fp16_ok=ok16)
        mode = "low-VRAM mode" if self.s.low_vram else "normal mode"
        out.append(Check("gpu", "GPU (main environment)", OK if ok32 else FAIL,
                         f"{props.name} · {data['vram_gb']} GB · compute {data['cc']} · torch {torch.__version__} "
                         f"· {mode} · {self.s.precision}", data=data))
        if not ok16 and self.s.precision == "fp16":
            out.append(Check("fp16", "Half precision", WARN, "fp16 math produced NaN/inf on this GPU",
                             "Settings → GPU → Precision: fp32."))
        return out

    def _env_check(self, cid: str, title: str, python: Path | None, import_code: str, packages: list[str],
                   cwd: Path | None, extra_path: list[Path], missing_fix: str) -> Check:
        if python is None:
            return Check(cid, title, WARN, "not installed", missing_fix)
        r = probe_env(python, import_code, packages, cwd=cwd, extra_path=extra_path)
        data = {"python_path": str(python), **r}
        if "error" in r:
            return Check(cid, title, FAIL, f"couldn't start: {r['error']}", "Re-run scripts\\setup_windows.ps1.", data)
        if not r.get("import_ok"):
            return Check(cid, title, FAIL, f"import failed: {r.get('import_error')}",
                         "Re-run scripts\\setup_windows.ps1 (the environment looks incomplete).", data)
        if "torch_error" in r:
            return Check(cid, title, FAIL, f"PyTorch broken: {r['torch_error']}", "Re-run scripts\\setup_windows.ps1.", data)
        main_cuda = self.s.gpu_info.cuda
        if main_cuda and not r.get("cuda_available"):
            why = "CPU-only PyTorch" if not r.get("torch_cuda") else "PyTorch can't see the GPU"
            return Check(cid, title, FAIL, f"{why} (torch {r.get('torch')}) while the main app has CUDA",
                         "Re-run scripts\\setup_windows.ps1; it reinstalls the CUDA build of PyTorch in this environment.",
                         data)
        dev = r.get("device") or "CPU"
        return Check(cid, title, OK, f"torch {r.get('torch')} · {dev}", data=data)

    def check_env_xtts(self) -> Check:
        from .tts.remote import REPO_ROOT, xtts_python

        return self._env_check("env_xtts", "XTTS environment (.venv-xtts)", xtts_python(),
                               "from TTS.tts.models.xtts import Xtts", ["coqui-tts", "transformers", "torch"],
                               None, [REPO_ROOT / "backend"],
                               "Optional. Re-run setup without -NoXtts to use the XTTS-v2 engine.")

    def check_env_vc(self) -> Check:
        from .vc.changer import REPO_ROOT, seedvc_root, vc_python

        root = seedvc_root()
        if vc_python() is not None and not (root / "modules").is_dir():
            return Check("env_vc", "Voice changer environment (.venv-vc)", FAIL, f"Seed-VC source missing at {root}",
                         "Re-run scripts\\setup_windows.ps1 (it downloads Seed-VC into vendor\\seed-vc).")
        return self._env_check("env_vc", "Voice changer environment (.venv-vc)", vc_python(),
                               "import sys; sys.path.insert(0, '.'); from modules.commons import build_model; import train",
                               ["torch", "transformers", "descript-audio-codec"], root, [REPO_ROOT / "backend", root],
                               "Optional. Re-run setup without -NoVc to use the real-time voice changer.")

    def check_audio(self) -> list[Check]:
        from .audio import devices as devmod
        from .audio.recorder import loopback_available

        ok, reason = devmod.available()
        if not ok:
            return [Check("audio", "Audio devices", FAIL, reason, "Reinstall the app (sounddevice/PortAudio).")]
        devs = devmod.list_devices()
        ins = [d for d in devs if d.max_input_channels > 0]
        outs = [d for d in devs if d.max_output_channels > 0]
        mic = devmod.default_device("input", devs)
        out = [Check("audio", "Audio devices", OK if ins and outs else WARN,
                     f"{len(ins)} inputs, {len(outs)} outputs ({devs[0].hostapi if devs else '?'}); "
                     f"default mic: {mic.name if mic else 'none'}",
                     "" if ins else "No microphone found: plug one in or enable it in Windows Sound settings.",
                     {"inputs": [d.name for d in ins], "outputs": [d.name for d in outs]})]
        virt = devmod.find_virtual_playback(devs)
        if virt:
            out.append(Check("vbcable", "Virtual cable", OK, f"found {virt.name}",
                             data={"device": virt.name}))
        else:
            out.append(Check("vbcable", "Virtual cable", WARN, "VB-Audio Virtual Cable not found",
                             "Install it from https://vb-audio.com/Cable/ (run as administrator), then reboot."))
        lok, lreason = loopback_available()
        out.append(Check("loopback", "Headphone recording", OK if lok else WARN,
                         "available" if lok else lreason, "" if lok else "pip install PyAudioWPatch (setup installs it)."))
        cfg = self.s.audio.config
        if cfg.output_device and not devmod.find_by_name(cfg.output_device, "output", devs):
            out.append(Check("audio_cfg", "Saved devices", WARN, f"saved output '{cfg.output_device}' is not connected",
                             "Pick it again on the Virtual mic page."))
        return out

    def check_engines(self) -> list[Check]:
        out = []
        cb = _pkg_version("chatterbox-tts")
        out.append(Check("chatterbox", "Chatterbox TTS", OK if cb else FAIL,
                         f"chatterbox-tts {cb}" if cb else "not installed",
                         "" if cb else "Re-run scripts\\setup_windows.ps1.",
                         {"transformers": _pkg_version("transformers")}))
        fw = _pkg_version("faster-whisper")
        if fw:
            cuda_n = 0
            try:
                import ctranslate2

                cuda_n = ctranslate2.get_cuda_device_count()
            except Exception:
                pass
            where = "CPU (low-VRAM mode)" if self.s.low_vram else ("GPU" if cuda_n else "CPU")
            out.append(Check("whisper", "Speech-to-text", OK, f"faster-whisper {fw} · will run on {where}",
                             data={"ctranslate2_cuda_devices": cuda_n}))
        else:
            out.append(Check("whisper", "Speech-to-text", WARN, "faster-whisper not installed",
                             "Optional. Re-run setup without -NoStt for the Record button."))
        return out  # XTTS / Seed-VC are covered by their environment checks

    def check_models(self) -> Check:
        """Which models this setup will use are already on disk (missing ones download on first use)."""
        from .tts.remote import xtts_python
        from .vc.changer import seedvc_root, vc_python

        cache = hf_cache_dir()
        items: list[tuple[str, bool, float]] = []
        if _pkg_version("chatterbox-tts"):
            items += [(label, hf_cached(repo, cache), gb) for label, repo, gb in CHATTERBOX_MODELS]
        if _pkg_version("faster-whisper"):
            label, repo, gb = WHISPER_CPU if (self.s.low_vram or not self.s.gpu_info.cuda) else WHISPER_GPU
            items.append((label, hf_cached(repo, cache), gb))
        if xtts_python() is not None:
            items.append(("XTTS-v2", (xtts_model_dir() / "model.pth").exists(), 1.9))
        if vc_python() is not None:
            ck = seedvc_root() / "checkpoints"
            items.append(("Seed-VC", ck.is_dir() and any(ck.glob("models--Plachta--Seed-VC*")), 0.4))
            items.append((VC_ENCODER[0], hf_cached(VC_ENCODER[1], cache), VC_ENCODER[2]))
        if not items:
            return Check("models", "Model downloads", SKIP, "no voice engines installed yet")
        missing = [(label, gb) for label, have, gb in items if not have]
        data = {"downloaded": [label for label, have, _ in items if have], "missing": [m[0] for m in missing]}
        if not missing:
            return Check("models", "Model downloads", OK, f"all {len(items)} models downloaded", data=data)
        need = sum(gb for _, gb in missing)
        free = shutil.disk_usage(self.s.paths.root).free / 1024**3
        detail = (f"{len(items) - len(missing)} of {len(items)} downloaded; first use will download "
                  f"{', '.join(m[0] for m in missing)} (~{need:.1f} GB)")
        if free < need + 2:
            return Check("models", "Model downloads", WARN, detail, f"Only {free:.1f} GB free: make room first.", data)
        return Check("models", "Model downloads", OK, detail, "A deep check downloads them now.", data)

    def check_network(self) -> list[Check]:
        targets = {"huggingface": "https://huggingface.co"}
        for pid, host in PROVIDER_HOSTS.items():
            if self.s.secrets.has(pid):
                targets[pid] = host

        def head(item):
            name, url = item
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "VoiceCloningTTS-doctor"})
                urllib.request.urlopen(req, timeout=8)
                code = 200
            except urllib.error.HTTPError as e:  # any HTTP answer means the host is reachable
                code = e.code
            except Exception as e:
                return name, url, None, str(getattr(e, "reason", e))
            return name, url, code, round((time.perf_counter() - t0) * 1000)

        out = []
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            for name, url, code, extra in pool.map(head, targets.items()):
                title = "Model download site (huggingface.co)" if name == "huggingface" else f"{name} API"
                if code is None:
                    out.append(Check(f"net:{name}", title, FAIL, f"can't reach {url}: {extra}",
                                     "Check your internet connection, firewall, VPN or proxy."))
                else:
                    out.append(Check(f"net:{name}", title, OK, f"reachable ({extra} ms)"))
        return out

    def check_keys(self) -> Check:
        configured = [p for p in ("openai", "anthropic", "gemini", "openrouter") if self.s.secrets.has(p)]
        persona = self.s.personas.resolve(None)
        data = {"configured": configured, "active_persona_provider": persona.provider,
                "active_persona_model": persona.model, "storage": self.s.secrets.backend_name}
        if not configured:
            return Check("keys", "API keys", WARN, "no LLM API key saved", "Settings → API keys.", data)
        if persona.provider not in configured:
            return Check("keys", "API keys", WARN, f"keys for {', '.join(configured)}, but the default persona uses "
                         f"{persona.provider}", "Add that key, or switch the persona's provider (Personas page).", data)
        return Check("keys", "API keys", OK, f"configured: {', '.join(configured)}", data=data)

    # -- deep checks -----------------------------------------------------------
    def _voice(self):
        s = self.s
        return s.voices.get(s.db.get_setting("active_voice_id")) or next(iter(s.voices.list()), None)

    async def deep_tts(self) -> Check:
        from .tts import CancelToken

        voice = self._voice()
        if voice is None:
            return Check("deep_tts", "Deep: speak with the cloned voice", SKIP, "no voice yet",
                         "Create a voice on the Voices page, then run the deep check again.")
        text = "This is a quick self test of the cloned voice. If you can hear me, the voice engine works."
        t0 = time.perf_counter()
        first = None
        chunks, sr = [], 24000
        async for audio, sr in self.s.tts.synthesize(text, voice, CancelToken()):
            first = first or time.perf_counter() - t0
            chunks.append(audio)
        total = time.perf_counter() - t0
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        dur = audio.size / sr if sr else 0
        if dur < 0.5 or not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 1e-3:
            return Check("deep_tts", "Deep: speak with the cloned voice", FAIL,
                         f"{voice.engine} produced no usable audio ({dur:.2f}s)",
                         "Switch precision to fp32 in Settings → GPU, or try the other engine.")
        self._tts_audio = (audio, sr)
        wav_path = None
        try:
            import soundfile as sf

            wav_path = self.s.paths.logs / "doctor-tts.wav"
            sf.write(wav_path, audio, sr)
        except Exception:
            pass
        rtf = total / dur
        status = OK if rtf < 1.0 else WARN
        return Check("deep_tts", "Deep: speak with the cloned voice", status,
                     f"{voice.engine} on '{voice.name}': {dur:.1f}s of speech in {total:.1f}s "
                     f"(first audio after {first:.1f}s, {rtf:.2f}× real time)",
                     "" if status == OK else "Slower than real time: replies will lag. Use Chatterbox Turbo for English, "
                     "or keep answers Short.",
                     {"engine": voice.engine, "rtf": round(rtf, 2), "first_audio_s": round(first or 0, 2),
                      "listen": str(wav_path) if wav_path else None})

    async def deep_vc(self) -> Check:
        from .vc import availability

        ok, reason = availability()
        if not ok:
            return Check("deep_vc", "Deep: voice changer speed", SKIP, reason)
        if self.s.vc.active:
            return Check("deep_vc", "Deep: voice changer speed", SKIP, "voice changer is running; stop it to benchmark")
        voice = self._voice()
        if voice is None:
            return Check("deep_vc", "Deep: voice changer speed", SKIP, "no voice yet", "Create a voice first.")
        s = self.s
        ref = s.voices.reference_path(voice.id)

        def run():
            s.claim_gpu("vc")
            s.vc.precision = s.precision
            t0 = time.perf_counter()
            r = s.vc.benchmark(ref, s.audio.config.sample_rate, s.vc.settings, s.voices.active_finetune(voice.id))
            r["load_s"] = round(time.perf_counter() - t0, 1)
            return r

        r = await asyncio.to_thread(run)
        load = r["load"]
        status = OK if load <= 0.75 else WARN if load <= 1.0 else FAIL
        fix = "" if status == OK else "Press ⚡ Auto-tune on the Virtual mic page (or pick the Low-end GPU preset)."
        return Check("deep_vc", "Deep: voice changer speed", status,
                     f"{r['infer_ms']:.0f} ms per {r['block_ms']:.0f} ms block ({load * 100:.0f}% load), "
                     f"~{r['latency_ms']:.0f} ms latency, {s.precision}", fix, r)

    async def deep_stt(self) -> Check:
        ok, reason = self.s.stt.availability()
        if not ok:
            return Check("deep_stt", "Deep: speech-to-text", SKIP, reason)
        if self._tts_audio is None:
            return Check("deep_stt", "Deep: speech-to-text", SKIP, "needs the TTS deep check to produce audio first")
        audio, sr = self._tts_audio
        if self.s.stt.uses_gpu():
            await asyncio.to_thread(self.s.gpu.acquire, "stt")
        t0 = time.perf_counter()
        r = await asyncio.to_thread(self.s.stt.transcribe, audio, sr, "en")
        took = time.perf_counter() - t0
        heard = r.get("text", "").strip()
        words = {"self", "test", "voice", "engine", "works"}
        hit = sum(1 for w in words if w in heard.lower())
        status = OK if hit >= 2 else WARN
        return Check("deep_stt", "Deep: speech-to-text", status,
                     f"heard \"{heard[:80]}\" in {took:.1f}s on {self.s.stt.last_device}",
                     "" if status == OK else "Transcript doesn't match well; the TTS voice may be unclear.",
                     {"seconds": round(took, 2), "device": self.s.stt.last_device})

    async def deep_llm(self) -> Check:
        from .llm import GenerationConfig, TextDelta
        from .llm.base import ChatMessage, ProviderError

        persona = self.s.personas.resolve(None)
        try:
            provider = self.s.providers.get(persona.provider)
        except ProviderError as e:
            return Check("deep_llm", "Deep: LLM reply", SKIP, str(e))
        cfg = GenerationConfig(model=persona.model, system_prompt="Reply with exactly: OK", max_tokens=16,
                               reasoning_effort="low" if persona.provider == "anthropic" else None)
        t0 = time.perf_counter()
        first = None
        text = ""
        try:
            async for ev in provider.stream([ChatMessage("user", "Say OK.")], cfg):
                if isinstance(ev, TextDelta):
                    first = first or time.perf_counter() - t0
                    text += ev.text
        except ProviderError as e:
            return Check("deep_llm", "Deep: LLM reply", FAIL, str(e), "Check the key and model on Settings / Personas.")
        took = time.perf_counter() - t0
        return Check("deep_llm", "Deep: LLM reply", OK if text.strip() else WARN,
                     f"{persona.provider}/{persona.model} replied \"{text.strip()[:40]}\" "
                     f"(first token {first or 0:.1f}s, total {took:.1f}s)",
                     data={"first_token_s": round(first or 0, 2), "total_s": round(took, 2)})


ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


def format_report(rep: dict) -> str:
    s = rep["summary"]
    lines = [
        f"VoiceCloningTTS self-check · app {rep['version']} · {rep['created_at']} · "
        f"{'deep' if rep['deep'] else 'quick'} ({rep['duration_s']}s)",
        f"{s['ok']} ok, {s['warn']} warnings, {s['fail']} failed, {s['skip']} skipped",
        "",
    ]
    for c in rep["checks"]:
        lines.append(f"{ICON.get(c['status'], c['status'])} {c['title']}: {c['detail']}")
        if c.get("fix") and c["status"] != OK:
            lines.append(f"       → {c['fix']}")
    pkgs = {p: _pkg_version(p) for p in ("torch", "torchaudio", "transformers", "chatterbox-tts",
                                         "faster-whisper", "ctranslate2", "sounddevice", "openai",
                                         "anthropic", "google-genai", "pywebview")}
    lines += ["", "Main environment packages: " + ", ".join(f"{k} {v}" for k, v in pkgs.items() if v)]
    for c in rep["checks"]:
        if c["id"] in ("env_xtts", "env_vc") and c.get("data", {}).get("packages"):
            lines.append(f"{c['title']} packages: " +
                         ", ".join(f"{k} {v}" for k, v in c["data"]["packages"].items()))
    return "\n".join(lines)


async def run_cli(deep: bool = False) -> int:
    """Entry point for `python -m vctts --doctor [--deep]` (no app server needed)."""
    from .state import AppState

    state = AppState()
    state.loop = asyncio.get_running_loop()

    def live(c: dict) -> None:
        print(f"{ICON.get(c['status'], c['status'])} {c['title']}: {c['detail']}", flush=True)

    print(f"Running {'deep' if deep else 'quick'} self-check…\n", flush=True)
    try:
        rep = await Doctor(state, on_result=live).run(deep)
    finally:
        try:
            state.vc.shutdown()
            state.tts.shutdown()
            state.db.close()
        except Exception:
            pass
    print("\n" + rep["text"])
    if rep.get("saved_to"):
        print(f"\nReport saved to {rep['saved_to']}")
    return 1 if rep["summary"]["fail"] else 0
