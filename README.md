# vLLM-SpecDec: High-Throughput LLM Serving with Speculative Decoding

A production-oriented starter project for speculative decoding in a vLLM fork, CUDA Graph capture planning for prefill/decode phases, multi-GPU tensor-parallel launch helpers, an MLPerf-style benchmark harness, and automated Nsight Systems bottleneck analysis.

> **Verification note:** this workspace includes runnable unit tests and synthetic benchmark generation. The `112 tokens/sec` and `92% GPU utilization` numbers are encoded as the **target A100 profile** in `benchmarks/target_a100_report.json`; they are not claimed as locally measured unless you run the benchmark on 2x A100 hardware.
>
> **Runtime note:** actual `vLLM` serving requires a **Linux/CUDA environment** with `PyTorch`, `vLLM`, and GPU drivers available. The current Windows workspace is suitable for scaffolding, tests, and packaging, but not for native A100 execution.

## ✅ Project Deliverables Covered

- **Speculative decoding overlay** for `Llama-3-8B`
- **CUDA Graph capture hooks** for `prefill` and `decode`
- **Tensor parallel launch plan** for **2x A100s**
- **MLPerf-style benchmark suite** for baseline vs optimized comparisons
- **Nsight Systems integration** for automatic bottleneck detection
- **Docker environment** for reproducible serving and profiling

## 📁 Repository Layout

```text
src/vllm_specdec/      Core optimizer modules
scripts/               Launch, benchmark, and profiling entrypoints
configs/               Llama-3-8B speculative decoding configs
benchmarks/            Target reports and benchmark docs
docker/                CUDA-ready container definition
tests/                 CPU-safe verification tests
```

## 🚀 Quick Start

### 1) Local verification

```powershell
python -m unittest discover -s tests -v
python scripts/benchmark.py --synthetic --output benchmarks/latest_report.json
```

### 2) Bootstrap and verify the upstream `vLLM` fork

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_vllm_fork.ps1
python scripts/vendor_smoke_check.py
```

### 3) Launch the optimized server on 2x A100s

```powershell
python scripts/launch_tp_server.py --show-runtime-plan
```

This emits a real upstream-compatible launch based on:

- `--speculative-config` for draft-model decoding
- `--compilation-config` for CUDA Graph capture sizes
- `--tensor-parallel-size 2` for the 2x A100 layout

### 4) Run an MLPerf-style comparison

```powershell
python scripts/benchmark.py --synthetic --output benchmarks/results/latest_report.json
```

For real hardware, use the commands from `configs/llama3_8b_specdec.json` or the output of `python scripts/vendor_smoke_check.py`.

### 5) Collect Nsight Systems traces

```powershell
python scripts/profile_nsys.py --summary-file .\profiles\sample_nsys_summary.txt
```

### 6) Run the full Project 1 validation pipeline

```powershell
python scripts/run_project1_pipeline.py --synthetic
```

### 7) Run a live serving benchmark on real A100 hardware

```powershell
python scripts/live_serving_benchmark.py --base-url http://127.0.0.1:8000 --label optimized-specdec
python scripts/compare_live_servers.py --baseline-url http://127.0.0.1:8001 --optimized-url http://127.0.0.1:8000
powershell -ExecutionPolicy Bypass -File .\scripts\run_real_a100_comparison.ps1
```

### 8) Use Docker Compose for the serving stack

The compose path now layers this repo on top of the official pre-built `vllm/vllm-openai:latest` image recommended by the vLLM Docker docs.

**Recommended live GPU path:** run this repo on a **Linux host with an NVIDIA GPU** and use the helper below:

```bash
chmod +x scripts/deploy_live_gpu_linux.sh
./scripts/deploy_live_gpu_linux.sh
```

For gated Llama checkpoints, export your Hugging Face token first:

```powershell
$env:HF_TOKEN = "<your_hf_token>"
docker builder prune -af   # optional, helps if earlier large pulls failed
docker compose -f .\docker\docker-compose.yml pull
docker compose -f .\docker\docker-compose.yml up
```

This starts:
- `vllm-baseline` on port `8001`
- `vllm-specdec` on port `8000`
- `compare-runner` to emit the live comparison report

> The first pull is large because `vllm/vllm-openai:latest` is a GPU runtime image. The stack mounts a persistent `hf-cache` volume and enables `ipc: host` for tensor-parallel serving.

## 📊 Target Performance Profile

| Metric | Baseline vLLM | Optimized target |
|---|---:|---:|
| Throughput | `45 tok/s` | `112 tok/s` |
| Speedup | `1.0x` | `2.49x` |
| GPU utilization | `68%` | `92%` |
| Memory footprint | `16.5 GB` | `13.53 GB` |
| Memory reduction | `0%` | `18%` |

## 🔧 Integration Strategy

1. Use `scripts/bootstrap_vllm_fork.ps1` to clone upstream vLLM into `vendor/vllm`.
2. Verify the upstream touchpoints with `scripts/vendor_smoke_check.py`.
3. Apply the runtime plan from `src/vllm_specdec/engine.py`, which now emits `--speculative-config` and `--compilation-config` for current vLLM.
4. Launch with the 2-GPU tensor-parallel plan from `src/vllm_specdec/tensor_parallel.py`.
5. Validate against the benchmark harness and profiler findings.

## 🧪 What is verified here

- Speculative token acceptance logic
- Benchmark speedup calculations
- Nsight summary parsing and bottleneck detection
- CLI generation for multi-GPU launch plans
- Upstream `vendor/vllm` compatibility probe

For real A100 numbers, run the provided benchmark and profiler scripts on the target hardware.
