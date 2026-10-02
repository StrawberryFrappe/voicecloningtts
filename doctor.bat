@echo off
rem Self-check: tests every part of the install and saves a report you can share.
rem   doctor.bat          quick check (seconds, downloads nothing)
rem   doctor.bat --deep   also loads the models and measures speed (downloads them the first time)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup_windows.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m vctts --doctor %*
echo.
echo The report was saved in %APPDATA%\VoiceCloningTTS\logs (doctor-*.txt). Paste it when asking for help.
pause
