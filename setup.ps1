param([string]$LaptopIp = '', [switch]$UseCuda)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 가상 환경 생성 실패' }
}
$requirementsFile = if ($UseCuda) { 'requirements-gpu.txt' } else { 'requirements.txt' }
& '.\.venv\Scripts\python.exe' -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw '패키지 설치 실패' }
$prepareDevice = if ($UseCuda) { 'cuda' } elseif ($env:LAVA_DEPTH_DEVICE) { $env:LAVA_DEPTH_DEVICE } else { 'cpu' }
if ($LaptopIp) {
    & '.\.venv\Scripts\python.exe' scripts/prepare.py --ip $LaptopIp --depth-device $prepareDevice
} else {
    & '.\.venv\Scripts\python.exe' scripts/prepare.py --depth-device $prepareDevice
}
if ($LASTEXITCODE -ne 0) { throw '모델·HTTPS 준비 실패' }
