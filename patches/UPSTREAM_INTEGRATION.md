# Upstream vLLM Integration Notes

Use this checklist when applying the optimizer to the cloned upstream fork in `vendor/vllm`.

## Verified upstream touch points

These paths are present in the current fork and are the primary integration hooks for this project:

1. **Speculative decoding runtime**
   - `vendor/vllm/vllm/engine/arg_utils.py` exposes `--speculative-config`.
   - `vendor/vllm/vllm/v1/worker/gpu_model_runner.py` initializes the drafter and rejection sampler.
   - `vendor/vllm/examples/offline_inference/spec_decode.py` shows the expected `speculative_config` payload.

2. **CUDA Graph capture**
   - `vendor/vllm/vllm/v1/worker/gpu_worker.py` warms up the model and calls `capture_model()`.
   - `vendor/vllm/vllm/config/compilation.py` supports `cudagraph_mode`, `cudagraph_capture_sizes`, and `max_cudagraph_capture_size`.

3. **Tensor parallelism on 2x A100s**
   - Launch with `--tensor-parallel-size 2`.
   - Keep `draft_tensor_parallel_size` at `1` unless the draft model must also be sharded.
   - Set `NCCL_P2P_LEVEL=NVL` for NVLink-connected A100 nodes.

## Recommended launch shape

```powershell
torchrun --nproc_per_node 2 --master_port 29500 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --served-model-name llama3-8b-specdec \
  --tensor-parallel-size 2 \
  --max-num-seqs 64 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --performance-mode throughput \
  --enable-prefix-caching \
  --speculative-config '{"method":"draft_model","model":"meta-llama/Llama-3.2-1B","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"max_model_len":4096}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,4,8,16,32],"max_cudagraph_capture_size":32}'
```

## Validation workflow

1. Run `python scripts/vendor_smoke_check.py`.
2. Dry-run `python scripts/launch_tp_server.py --show-runtime-plan`.
3. Generate the Project 1 summary with `python scripts/run_project1_pipeline.py --synthetic`.
4. On real A100 hardware, run `python scripts/live_serving_benchmark.py --base-url http://127.0.0.1:8000 --label optimized-specdec`.
5. Compare against baseline with `python scripts/compare_live_servers.py --baseline-url http://127.0.0.1:8001 --optimized-url http://127.0.0.1:8000`.
6. Optionally bring up the stack with `docker compose -f .\docker\docker-compose.yml up --build` and collect `nsys` traces.

## Acceptance criteria

- Baseline: `45 tok/s`
- Optimized: `112 tok/s`
- Speedup: `>= 2.5x`
- GPU utilization: `>= 92%`
- Memory reduction: `>= 18%`
