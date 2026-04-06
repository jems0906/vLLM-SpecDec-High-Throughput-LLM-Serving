param(
    [string]$RepoUrl = "https://github.com/vllm-project/vllm.git",
    [string]$TargetDir = "vendor/vllm"
)

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    if (-not (Test-Path $TargetDir)) {
        Write-Host "Cloning upstream vLLM into $TargetDir ..."
        git clone $RepoUrl $TargetDir
    } else {
        Write-Host "vLLM fork already exists at $TargetDir"
    }

    $notes = @"
Next steps for Project 1 integration:
1. Open src\vllm_specdec\engine.py and align the CLI flags with your target vLLM version.
2. Validate the CUDA graph capture plan from src\vllm_specdec\cuda_graphs.py.
3. Launch the 2x A100 tensor-parallel server with scripts\launch_tp_server.py.
4. Run scripts\benchmark.py and scripts\profile_nsys.py on real hardware.
"@

    $notesPath = Join-Path $root "vendor\integration-notes.txt"
    $notes | Set-Content -Path $notesPath -Encoding UTF8
    Write-Host "Integration notes written to $notesPath"
}
finally {
    Pop-Location
}
