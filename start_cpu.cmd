@echo off
setlocal
set "LAVA_DEPTH_DEVICE=cpu"
call "%~dp0start.cmd" %* --depth-device cpu
exit /b %errorlevel%
