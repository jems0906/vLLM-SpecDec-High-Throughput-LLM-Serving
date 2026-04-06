from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.live_benchmark import ServingBenchmarkSummary, load_prompts, run_live_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live OpenAI-compatible serving benchmark against vLLM.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--prompts-file", default=str(ROOT / "benchmarks" / "prompts.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--gpu-utilization", type=float, default=92.0)
    parser.add_argument("--memory-gb", type=float, default=13.53)
    parser.add_argument("--label", default="optimized-specdec")
    parser.add_argument("--simulate", action="store_true", help="Emit a local sample summary without contacting a server.")
    return parser.parse_args()


def emit(summary: ServingBenchmarkSummary) -> int:
    for line in summary.to_console_lines():
        print(line)
    return 0


def main() -> int:
    args = parse_args()
    if args.simulate:
        summary = ServingBenchmarkSummary(
            label=args.label,
            tokens_per_sec=112.0,
            gpu_utilization=args.gpu_utilization,
            memory_gb=args.memory_gb,
            p50_latency_ms=34.0,
            completed_requests=32,
            total_tokens=3584,
        )
        return emit(summary)

    prompts = load_prompts(args.prompts_file)
    summary = run_live_benchmark(
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        timeout=args.timeout,
        gpu_utilization=args.gpu_utilization,
        memory_gb=args.memory_gb,
        label=args.label,
    )
    return emit(summary)


if __name__ == "__main__":
    raise SystemExit(main())
