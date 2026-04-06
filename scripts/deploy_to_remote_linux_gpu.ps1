[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$UserName,

    [string]$RemoteDir = "~/vllm-specdec",
    [string]$HFToken,
    [string]$KeyFile,
    [switch]$SkipUpload,
    [switch]$SkipRun
)

$ErrorActionPreference = 'Stop'

function Assert-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on this machine."
    }
}

function Invoke-External([string]$FilePath, [string[]]$Arguments) {
    Write-Host "> $FilePath $($Arguments -join ' ')" -ForegroundColor Cyan
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

Assert-Command ssh
Assert-Command scp

$root = Split-Path -Parent $PSScriptRoot
$artifactDir = Join-Path $root '.artifacts'
$archivePath = Join-Path $artifactDir 'vllm-specdec-deploy.zip'

New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
if (Test-Path $archivePath) {
    Remove-Item $archivePath -Force
}

$stagingDir = Join-Path $artifactDir 'staging'
if (Test-Path $stagingDir) {
    Remove-Item $stagingDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

$excludeNames = @('.git', '.venv', '.artifacts', '__pycache__')
Get-ChildItem -Force $root | Where-Object { $excludeNames -notcontains $_.Name } | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $stagingDir -Recurse -Force
}

Compress-Archive -Path (Join-Path $stagingDir '*') -DestinationPath $archivePath -Force

$sshTarget = "$UserName@$HostName"
$scpArgs = @()
$sshArgs = @()
if ($KeyFile) {
    $scpArgs += @('-i', $KeyFile)
    $sshArgs += @('-i', $KeyFile)
}

if (-not $SkipUpload) {
    Invoke-External scp ($scpArgs + @($archivePath, "${sshTarget}:~/vllm-specdec-deploy.zip"))
}

$escapedRemoteDir = $RemoteDir.Replace("'", "'\''")
$remoteLines = @(
    'set -euo pipefail',
    "mkdir -p '$escapedRemoteDir'",
    "if command -v unzip >/dev/null 2>&1; then unzip -oq ~/vllm-specdec-deploy.zip -d '$escapedRemoteDir'; else python3 - <<'" + 'PY' + "'\nimport zipfile\nzipfile.ZipFile('/home/' + __import__('os').environ.get('USER','') + '/vllm-specdec-deploy.zip').extractall('$escapedRemoteDir')\nPY\nfi",
    "cd '$escapedRemoteDir'",
    "if [ ! -f docker/.env ]; then cp docker/.env.example docker/.env; fi"
)

if ($HFToken) {
    $escapedToken = $HFToken.Replace("'", "'\''")
    $remoteLines += "python3 - <<'PY'\nfrom pathlib import Path\np = Path('docker/.env')\ntext = p.read_text()\nlines = []\nreplaced = False\nfor line in text.splitlines():\n    if line.startswith('HF_TOKEN='):\n        lines.append('HF_TOKEN=$escapedToken')\n        replaced = True\n    else:\n        lines.append(line)\nif not replaced:\n    lines.append('HF_TOKEN=$escapedToken')\np.write_text('\\n'.join(lines) + '\\n')\nPY"
}

$remoteLines += @(
    'chmod +x scripts/deploy_live_gpu_linux.sh',
    './scripts/deploy_live_gpu_linux.sh'
)

$remoteScript = ($remoteLines -join "`n")
$tempRemoteScript = Join-Path $artifactDir 'remote-deploy.sh'
Set-Content -Path $tempRemoteScript -Value $remoteScript -NoNewline

if (-not $SkipRun) {
    Get-Content -Raw $tempRemoteScript | & ssh @sshArgs $sshTarget 'bash -s'
    if ($LASTEXITCODE -ne 0) {
        throw "Remote deployment failed with exit code $LASTEXITCODE."
    }
}

Write-Host "`nRemote deployment package ready: $archivePath" -ForegroundColor Green
Write-Host "Target: $sshTarget" -ForegroundColor Green
Write-Host 'Done.' -ForegroundColor Green
