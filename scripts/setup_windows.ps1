<#
  VoiceCloningTTS - one-time setup for Windows (NVIDIA GPU).

  Run from the repo root in PowerShell:
      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

  Options:
      -Cpu          install CPU-only PyTorch (no NVIDIA GPU)
      -NoXtts       skip XTTS-v2 (Coqui) engine
      -NoStt        skip speech-to-text (faster-whisper)
      -NoVc         skip the real-time voice changer (Seed-VC)
#>
param(
    [switch]$Cpu,
    [switch]$NoXtts,
    [switch]$NoStt,
    [switch]$NoVc
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Python 3.11 ------------------------------------------------------------
Step "Checking Python 3.11"
$py = $null
try { & py -3.11 --version *> $null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-3.11") } } catch {}
if (-not $py) {
    Fail "Python 3.11 not found. Install it with:  winget install Python.Python.3.11   (then re-run this script)"
}

# --- Node.js (to build the UI) ------------------------------------------------
Step "Checking Node.js"
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail "Node.js not found. Install it with:  winget install OpenJS.NodeJS.LTS   (then open a new terminal and re-run)"
}

# --- Virtual environment ----------------------------------------------------
Step "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $py[0] $py[1] -m venv .venv
}
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel setuptools

# --- PyTorch ----------------------------------------------------------------
if ($Cpu) {
    Step "Installing PyTorch 2.6 (CPU)"
    & $vpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
} else {
    Step "Installing PyTorch 2.6 (CUDA 12.4)"
    & $vpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
}
if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed" }

# --- App + engines ------------------------------------------------------------
$extras = @("desktop", "chatterbox", "loopback")
if (-not $NoStt) { $extras += "stt" }
$spec = "backend[" + ($extras -join ",") + "]"
Step "Installing VoiceCloningTTS ($spec)"
& $vpy -m pip install -e $spec
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }

# Make sure pip didn't swap the CUDA build for a CPU one while resolving deps.
if (-not $Cpu) {
    $cuda = & $vpy -c "import torch; print(torch.cuda.is_available())"
    if ($cuda -ne "True") {
        Write-Host "PyTorch can't see the GPU; reinstalling the CUDA build..." -ForegroundColor Yellow
        & $vpy -m pip install --force-reinstall --no-deps torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
        $cuda = & $vpy -c "import torch; print(torch.cuda.is_available())"
    }
    Write-Host "CUDA available: $cuda"
}

# --- XTTS-v2 in its own environment ---------------------------------------------
# chatterbox-tts and coqui-tts need incompatible `transformers` versions, so XTTS
# lives in .venv-xtts and the app runs it as a background worker process.
if (-not $NoXtts) {
    Step "Installing XTTS-v2 into .venv-xtts"
    if (-not (Test-Path ".venv-xtts\Scripts\python.exe")) {
        & $py[0] $py[1] -m venv .venv-xtts
    }
    $xpy = Join-Path $Root ".venv-xtts\Scripts\python.exe"
    & $xpy -m pip install --upgrade pip wheel setuptools
    if ($Cpu) {
        & $xpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    } else {
        & $xpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    }
    & $xpy -m pip install "coqui-tts==0.27.5" "transformers>=4.57,<5" pydantic
    if ($LASTEXITCODE -ne 0) { Fail "XTTS install failed (re-run with -NoXtts to skip it)" }
    & $xpy -m pip install --no-deps -e backend
    & $xpy -c "from TTS.tts.models.xtts import Xtts; print('XTTS OK')"
    if ($LASTEXITCODE -ne 0) { Fail "XTTS environment check failed" }
}

# --- Real-time voice changer (Seed-VC) in its own environment ---------------------
# Seed-VC (GPL-3.0) isn't on PyPI: download a pinned snapshot into vendor\seed-vc
# and give it its own .venv-vc with the versions it was built against.
if (-not $NoVc) {
    $SeedCommit = "51383efd921027683c89e5348211d93ff12ac2a8"
    $SeedDir = Join-Path $Root "vendor\seed-vc"
    if (-not (Test-Path (Join-Path $SeedDir "modules"))) {
        Step "Downloading Seed-VC ($($SeedCommit.Substring(0,7)))"
        New-Item -ItemType Directory -Force -Path (Join-Path $Root "vendor") | Out-Null
        $zip = Join-Path $env:TEMP "seed-vc.zip"
        Invoke-WebRequest -Uri "https://github.com/Plachtaa/seed-vc/archive/$SeedCommit.zip" -OutFile $zip
        $tmp = Join-Path $env:TEMP "seed-vc-extract"
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
        Expand-Archive -Path $zip -DestinationPath $tmp
        if (Test-Path $SeedDir) { Remove-Item -Recurse -Force $SeedDir }
        Move-Item (Join-Path $tmp "seed-vc-$SeedCommit") $SeedDir
        Remove-Item -Recurse -Force $tmp, $zip
    }
    Step "Installing the voice changer into .venv-vc"
    if (-not (Test-Path ".venv-vc\Scripts\python.exe")) {
        & $py[0] $py[1] -m venv .venv-vc
    }
    $cpy = Join-Path $Root ".venv-vc\Scripts\python.exe"
    & $cpy -m pip install --upgrade pip wheel setuptools
    if ($Cpu) {
        & $cpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    } else {
        & $cpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    }
    & $cpy -m pip install "transformers==4.46.3" "librosa==0.10.2" "munch==4.0.0" "einops==0.8.0" `
        "descript-audio-codec==1.0.0" "huggingface-hub>=0.28.1" pyyaml soundfile scipy "numpy==1.26.4" pydantic
    if ($LASTEXITCODE -ne 0) { Fail "Voice changer install failed (re-run with -NoVc to skip it)" }
    & $cpy -m pip install --no-deps -e backend
    Push-Location $SeedDir
    & $cpy -c "from modules.commons import build_model; from modules.hifigan.generator import HiFTGenerator; print('Seed-VC OK')"
    $ok = $LASTEXITCODE
    Pop-Location
    if ($ok -ne 0) { Fail "Voice changer environment check failed" }
}

# --- UI -----------------------------------------------------------------------
Step "Building the UI"
Push-Location frontend
npm install --no-audit --no-fund
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "npm install failed" }
npm run build
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "UI build failed" }
Pop-Location

# --- Virtual cable check ------------------------------------------------------
Step "Checking for a virtual audio cable"
$cable = & $vpy -c "from vctts.audio.devices import virtual_cable_status as s; print(s().get('device') or '')"
if ($cable) {
    Write-Host "Found: $cable" -ForegroundColor Green
} else {
    Write-Host "VB-Audio Virtual Cable not found." -ForegroundColor Yellow
    Write-Host "  1. Download it from https://vb-audio.com/Cable/"
    Write-Host "  2. Run VBCABLE_Setup_x64.exe as administrator -> Install Driver, then reboot."
}

# --- Desktop shortcut -----------------------------------------------------------
Step "Creating desktop shortcut"
try {
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "VoiceCloningTTS.lnk"))
    $lnk.TargetPath = Join-Path $Root "run.bat"
    $lnk.WorkingDirectory = $Root
    $lnk.WindowStyle = 7
    $lnk.Save()
} catch { Write-Host "(could not create shortcut: $_)" }

Write-Host "`nDone! Start the app with run.bat (or the desktop shortcut)." -ForegroundColor Green
Write-Host "The first time you use a voice engine it downloads its model (a few GB)."
