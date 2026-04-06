param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$BaselineUrl = "http://127.0.0.1:8001",
    [string]$Model = "meta-llama/Meta-Llama-3-8B-Instruct",
    [switch]$CompareOnly
)

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not $CompareOnly) {
    Write-Host "1) Verify workspace"
    & $python -m unittest discover -s (Join-Path $root "tests") -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "2) Probe vendor vLLM"
    & $python (Join-Path $root "scripts\vendor_smoke_check.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "3) Run optimized live benchmark against $BaseUrl"
    & $python (Join-Path $root "scripts\live_serving_benchmark.py") --base-url $BaseUrl --model $Model --label optimized-specdec
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "4) Compare baseline and optimized servers"
& $python (Join-Path $root "scripts\compare_live_servers.py") --baseline-url $BaselineUrl --optimized-url $BaseUrl --model $Model
