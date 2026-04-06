# MLPerf-style Benchmark Suite

This folder stores reproducible benchmark inputs and outputs for comparing:

- `baseline-vllm`
- `optimized-specdec`

## Metrics tracked

- tokens/sec
- P50 latency
- GPU utilization
- memory footprint
- achieved speedup

## Suggested real-hardware flow

```powershell
python scripts/benchmark.py \
  --baseline-cmd "python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 2" \
  --optimized-cmd "python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 2 --speculative-config {\"method\":\"draft_model\",\"model\":\"meta-llama/Llama-3.2-1B\",\"num_speculative_tokens\":5,\"draft_tensor_parallel_size\":1,\"max_model_len\":4096} --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,4,8,16,32],\"max_cudagraph_capture_size\":32}" \
  --output benchmarks/results/a100_run.json
```

Use `target_a100_report.json` as the target acceptance profile for Project 1.

For a real API benchmark, start both baseline and optimized servers and run:

```powershell
python scripts/live_serving_benchmark.py --base-url http://127.0.0.1:8000 --prompts-file benchmarks/prompts.jsonl --label optimized-specdec
python scripts/compare_live_servers.py --baseline-url http://127.0.0.1:8001 --optimized-url http://127.0.0.1:8000
```
