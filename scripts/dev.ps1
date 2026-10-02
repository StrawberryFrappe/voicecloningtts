# Development mode: backend with auto-reload on :8765 + Vite dev server on :5173.
$Root = Split-Path -Parent $PSScriptRoot
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
Start-Process -NoNewWindow $vpy -ArgumentList "-m", "uvicorn", "vctts.main:create_app", "--factory", "--port", "8765", "--reload", "--app-dir", (Join-Path $Root "backend")
Set-Location (Join-Path $Root "frontend")
npm run dev
