from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.benchmark import BenchmarkSuite
from vllm_specdec.config import OptimizerConfig
from vllm_specdec.nsys_profiler import NsightAnalyzer
from vllm_specdec.reporting import ProjectRunSummary
from vllm_specdec.vendor_integration import VendorVllmIntegrationProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Project 1 validation pipeline.")
    parser.add_argument("--synthetic", action="store_true", help="Use target A100 benchmark values instead of executing live server commands.")
    parser.add_argument("--baseline-cmd", help="Optional live baseline benchmark command.")
    parser.add_argument("--optimized-cmd", help="Optional live optimized benchmark command.")
    parser.add_argument("--nsys-summary-file", default=str(ROOT / "profiles" / "sample_nsys_summary.txt"))
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks" / "results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OptimizerConfig.from_json(ROOT / "configs" / "llama3_8b_specdec.json")
    vendor_root = (ROOT / config.vendor_vllm_dir).resolve() if not config.vendor_vllm_dir.is_absolute() else config.vendor_vllm_dir
    vendor_summary = VendorVllmIntegrationProbe(vendor_root).summary()

    if args.synthetic or not (args.baseline_cmd and args.optimized_cmd):
        benchmark = BenchmarkSuite.synthetic_target()
    else:
        suite = BenchmarkSuite()
        baseline = suite.run_command(args.baseline_cmd, "baseline-vllm")
        optimized = suite.run_command(args.optimized_cmd, "optimized-specdec")
        benchmark = suite.compare(baseline, optimized)

    findings = NsightAnalyzer().analyze_file(args.nsys_summary_file)
    summary = ProjectRunSummary(vendor_summary=vendor_summary, benchmark=benchmark, findings=findings)
    json_path, md_path = summary.write(args.output_dir)

    print("=== Project 1 Summary ===")
    print(summary.to_markdown())
    print(f"\nSaved JSON summary to: {json_path}")
    print(f"Saved Markdown summary to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
