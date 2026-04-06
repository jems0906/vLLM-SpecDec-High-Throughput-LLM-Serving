from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.benchmark import BenchmarkSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline vs optimized benchmark comparison.")
    parser.add_argument("--baseline-cmd", help="Command that prints tokens/sec, GPU util, memory, and latency.")
    parser.add_argument("--optimized-cmd", help="Command that prints tokens/sec, GPU util, memory, and latency.")
    parser.add_argument("--synthetic", action="store_true", help="Emit the target A100 comparison profile without running commands.")
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" / "latest_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.synthetic or not (args.baseline_cmd and args.optimized_cmd):
        comparison = BenchmarkSuite.synthetic_target()
    else:
        suite = BenchmarkSuite()
        baseline = suite.run_command(args.baseline_cmd, "baseline-vllm")
        optimized = suite.run_command(args.optimized_cmd, "optimized-specdec")
        comparison = suite.compare(baseline, optimized)

    output_path = BenchmarkSuite.write_report(comparison, args.output)
    print(BenchmarkSuite.format_markdown(comparison))
    print(f"\nSaved report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
