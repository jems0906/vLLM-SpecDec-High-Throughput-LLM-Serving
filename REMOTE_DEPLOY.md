# Remote Linux/NVIDIA Deployment

Use this when you want to push the repo from the current Windows workspace to a remote Linux host with an NVIDIA GPU and start the live stack over SSH.

## Requirements
- `ssh` and `scp` available on the Windows machine
- remote host already has Docker, Docker Compose, NVIDIA drivers, and the NVIDIA container runtime
- a valid Hugging Face token for gated model access

## One-command deployment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_remote_linux_gpu.ps1 `
  -HostName <server-or-ip> `
  -UserName <ssh-user> `
  -RemoteDir ~/vllm-specdec `
  -HFToken <your_hf_token>
```

## Optional SSH key

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_remote_linux_gpu.ps1 `
  -HostName <server-or-ip> `
  -UserName <ssh-user> `
  -KeyFile C:\Users\you\.ssh\id_rsa `
  -HFToken <your_hf_token>
```

## What it does
1. creates a deployment zip from this repo
2. uploads it with `scp`
3. unpacks it on the remote Linux host
4. ensures `docker/.env` exists
5. injects `HF_TOKEN` if provided
6. runs `scripts/deploy_live_gpu_linux.sh`

## Notes
- This script cannot bypass missing SSH/network access.
- If the remote host is not already Docker/NVIDIA-ready, prepare that first.
