@echo off
setlocal
set PYTHONUTF8=1
chcp 65001 >nul
cd /d "%~dp0"
title AR Floor Is Lava

if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, aiortc, cv2, mediapipe, onnxruntime, cryptography" >nul 2>&1
if errorlevel 1 goto setup
goto prepare

:setup
echo Preparing the Python environment. The first run may take a few minutes.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 goto failed

:prepare
".venv\Scripts\python.exe" scripts\prepare.py
if errorlevel 1 goto failed
echo Starting the server. Your browser will open when it is ready.
echo Keep this window open. Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" scripts\run.py --open-browser %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Startup failed. Check the error above.
pause
exit /b 1
