$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' scripts/run.py
if ($LASTEXITCODE -ne 0) { throw '서버 실행 실패' }
