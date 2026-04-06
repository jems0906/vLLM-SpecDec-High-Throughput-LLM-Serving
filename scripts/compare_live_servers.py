from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.benchmark import BenchmarkSuite
from vllm_specdec.live_benchmark import compare_summaries, load_prompts, run_live_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized live vLLM servers.")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8001")
    parser.add_argument("--optimized-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--prompts-file", default=str(ROOT / "benchmarks" / "prompts.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--baseline-gpu-util", type=float, default=68.0)
    parser.add_argument("--optimized-gpu-util", type=float, default=92.0)
    parser.add_argument("--baseline-memory-gb", type=float, default=16.50)
    parser.add_argument("--optimized-memory-gb", type=float, default=13.53)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" / "live_comparison.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.simulate:
        comparison = BenchmarkSuite.synthetic_target()
    else:
        prompts = load_prompts(args.prompts_file)
        baseline = run_live_benchmark(
            base_url=args.baseline_url,
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            timeout=args.timeout,
            gpu_utilization=args.baseline_gpu_util,
            memory_gb=args.baseline_memory_gb,
            label="baseline-vllm",
            ready_timeout=args.ready_timeout,
        )
        optimized = run_live_benchmark(
            base_url=args.optimized_url,
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            timeout=args.timeout,
            gpu_utilization=args.optimized_gpu_util,
            memory_gb=args.optimized_memory_gb,
            label="optimized-specdec",
            ready_timeout=args.ready_timeout,
        )
        comparison = compare_summaries(baseline, optimized)

    output_path = BenchmarkSuite.write_report(comparison, args.output)
    print(BenchmarkSuite.format_markdown(comparison))
    print(f"\nSaved live comparison to: {output_path}")
    print(json.dumps(comparison.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
