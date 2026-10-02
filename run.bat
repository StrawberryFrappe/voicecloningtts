@echo off
rem Launch VoiceCloningTTS (desktop window). Logs: %APPDATA%\VoiceCloningTTS\logs\vctts.log
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run scripts\setup_windows.ps1 first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m vctts %*
