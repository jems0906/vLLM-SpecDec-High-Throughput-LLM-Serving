from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.nsys_profiler import NsightAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an Nsight Systems summary or collect one with nsys.")
    parser.add_argument("--summary-file", help="Existing Nsight summary text file to analyze.")
    parser.add_argument("--command", help="Optional workload command to profile with nsys.")
    parser.add_argument("--output", default=str(ROOT / "profiles" / "vllm_specdec"))
    return parser.parse_args()


def maybe_profile_with_nsys(command: str, output_prefix: str) -> Path:
    if shutil.which("nsys") is None:
        raise RuntimeError("nsys is not installed or not on PATH. Use --summary-file to analyze an exported summary.")
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nsys_command = f'nsys profile --stats=true --force-overwrite true -o "{output_path}" {command}'
    completed = subprocess.run(nsys_command, shell=True, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    summary_file = output_path.with_suffix(".txt")
    summary_file.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return summary_file


def main() -> int:
    args = parse_args()
    analyzer = NsightAnalyzer()

    if args.command:
        summary_path = maybe_profile_with_nsys(args.command, args.output)
    elif args.summary_file:
        summary_path = Path(args.summary_file)
    else:
        raise SystemExit("Pass either --summary-file or --command")

    findings = analyzer.analyze_file(summary_path)
    print(f"Analyzed: {summary_path}")
    for finding in findings:
        print(f"- [{finding.severity}] {finding.category}: {finding.evidence}")
        print(f"  recommendation: {finding.recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
