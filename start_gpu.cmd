@echo off
setlocal
set "LAVA_DEPTH_DEVICE=cuda"
call "%~dp0start.cmd" %* --depth-device cuda
exit /b %errorlevel%
