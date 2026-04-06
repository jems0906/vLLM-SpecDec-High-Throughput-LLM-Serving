# Project 1 Runbook

This repository is prepared for **Project 1: `vLLM-SpecDec`** and includes both a synthetic verification path and a real 2x A100 execution path.

> **Prerequisite:** run the real serving path on a **Linux/CUDA host** or through `docker compose` with the NVIDIA container runtime. Native Windows execution in this workspace does not currently include `torch`/`vllm`.
>
> **Best option for live GPU:** clone or copy this repo to a Linux host with an **NVIDIA GPU**, set `docker/.env`, and run:
>
> ```bash
> chmod +x scripts/deploy_live_gpu_linux.sh
> ./scripts/deploy_live_gpu_linux.sh
> ```

## 1. Verified local checks

Run these first:

```powershell
python -m unittest discover -s tests -v
python scripts/vendor_smoke_check.py
python scripts/run_project1_pipeline.py --synthetic
```

## 2. Launch sequence on real hardware

### Terminal A — baseline server
```powershell
$env:PYTHONPATH = "$PWD\vendor\vllm"
python -m vllm.entrypoints.openai.api_server `
  --host 0.0.0.0 `
  --port 8001 `
  --model meta-llama/Meta-Llama-3-8B-Instruct `
  --served-model-name llama3-8b-baseline `
  --tensor-parallel-size 2 `
  --max-num-seqs 64 `
  --max-model-len 4096 `
  --gpu-memory-utilization 0.92 `
  --disable-log-stats
```

### Terminal B — optimized speculative-decoding server
```powershell
python scripts/launch_tp_server.py --execute
```

### Terminal C — live comparison
```powershell
python scripts/compare_live_servers.py `
  --baseline-url http://127.0.0.1:8001 `
  --optimized-url http://127.0.0.1:8000
```

## 3. Profile with Nsight Systems

```powershell
python scripts/profile_nsys.py --summary-file .\profiles\sample_nsys_summary.txt
```

For a real run, replace the sample summary with an exported `nsys` text report.

## 4. Docker workflow

The compose stack uses the official pre-built `vllm/vllm-openai:latest` image as the runtime base, which is faster and more reliable than installing `torch` and `vllm` from scratch inside the container.

```powershell
$env:HF_TOKEN = "<your_hf_token>"
docker builder prune -af   # optional cleanup if earlier pulls/builds failed
docker compose -f .\docker\docker-compose.yml pull
docker compose -f .\docker\docker-compose.yml up
```

On the recommended Linux/NVIDIA host, the fastest path is:

```bash
chmod +x scripts/deploy_live_gpu_linux.sh
./scripts/deploy_live_gpu_linux.sh
```

This starts:
- `vllm-baseline`
- `vllm-specdec`
- `compare-runner`

> Use this on a Linux/WSL2 CUDA host with the NVIDIA container runtime enabled. The stack shares Hugging Face cache data through the `hf-cache` Docker volume and now avoids local image builds.

## 5. Expected acceptance targets

- Baseline: `45 tok/s`
- Optimized: `112 tok/s`
- Speedup: `>= 2.5x`
- GPU utilization: `>= 92%`
- Memory reduction: `>= 18%`

## 6. Output artifacts

Generated files are written under `benchmarks/results/`:
- `latest_report.json`
- `live_comparison.json`
- `project1_summary.json`
- `project1_summary.md`
