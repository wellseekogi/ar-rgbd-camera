param([string]$LaptopIp = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 가상 환경 생성 실패' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '패키지 설치 실패' }
if ($LaptopIp) {
    & '.\.venv\Scripts\python.exe' scripts/prepare.py --ip $LaptopIp
} else {
    & '.\.venv\Scripts\python.exe' scripts/prepare.py
}
if ($LASTEXITCODE -ne 0) { throw '모델·HTTPS 준비 실패' }
