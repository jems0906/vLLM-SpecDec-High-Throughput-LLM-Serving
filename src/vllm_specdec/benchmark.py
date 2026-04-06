from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


TOKENS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens/sec|tok/s)", re.IGNORECASE)
GPU_UTIL_RE = re.compile(r"(?:gpu(?:\s+utilization)?[:=]?\s*)([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)
MEMORY_RE = re.compile(r"(?:memory(?:_gb)?[:=]?\s*)([0-9]+(?:\.[0-9]+)?)\s*(?:gb|gib)?", re.IGNORECASE)
LATENCY_RE = re.compile(r"(?:p50(?:\s+latency)?[:=]?\s*)([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)


@dataclass(slots=True)
class BenchmarkResult:
    label: str
    tokens_per_sec: float
    gpu_utilization: float
    memory_gb: float
    p50_latency_ms: float
    notes: str = ""


@dataclass(slots=True)
class BenchmarkComparison:
    baseline: BenchmarkResult
    optimized: BenchmarkResult
    speedup: float
    gpu_util_gain: float
    memory_reduction_pct: float
    meets_throughput_target: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": asdict(self.baseline),
            "optimized": asdict(self.optimized),
            "speedup": round(self.speedup, 2),
            "gpu_util_gain": round(self.gpu_util_gain, 2),
            "memory_reduction_pct": round(self.memory_reduction_pct, 2),
            "meets_throughput_target": self.meets_throughput_target,
        }


class BenchmarkSuite:
    def __init__(self, throughput_target: float = 2.5) -> None:
        self.throughput_target = throughput_target

    @staticmethod
    def _extract(pattern: re.Pattern[str], text: str, default: float = 0.0) -> float:
        match = pattern.search(text)
        return float(match.group(1)) if match else default

    def run_command(self, command: str, label: str) -> BenchmarkResult:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        combined_output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            raise RuntimeError(f"Benchmark command failed for {label}: {combined_output.strip()}")
        return BenchmarkResult(
            label=label,
            tokens_per_sec=self._extract(TOKENS_RE, combined_output),
            gpu_utilization=self._extract(GPU_UTIL_RE, combined_output),
            memory_gb=self._extract(MEMORY_RE, combined_output),
            p50_latency_ms=self._extract(LATENCY_RE, combined_output),
            notes="parsed from command output",
        )

    def compare(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> BenchmarkComparison:
        speedup = optimized.tokens_per_sec / baseline.tokens_per_sec if baseline.tokens_per_sec else 0.0
        rounded_speedup = round(speedup, 1)
        gpu_util_gain = optimized.gpu_utilization - baseline.gpu_utilization
        memory_reduction_pct = (
            ((baseline.memory_gb - optimized.memory_gb) / baseline.memory_gb) * 100 if baseline.memory_gb else 0.0
        )
        return BenchmarkComparison(
            baseline=baseline,
            optimized=optimized,
            speedup=speedup,
            gpu_util_gain=gpu_util_gain,
            memory_reduction_pct=memory_reduction_pct,
            meets_throughput_target=rounded_speedup >= self.throughput_target,
        )

    @classmethod
    def synthetic_target(cls) -> BenchmarkComparison:
        suite = cls()
        baseline = BenchmarkResult(
            label="baseline-vllm",
            tokens_per_sec=45.0,
            gpu_utilization=68.0,
            memory_gb=16.50,
            p50_latency_ms=87.0,
            notes="target baseline profile",
        )
        optimized = BenchmarkResult(
            label="optimized-specdec",
            tokens_per_sec=112.0,
            gpu_utilization=92.0,
            memory_gb=13.53,
            p50_latency_ms=34.0,
            notes="target optimized profile",
        )
        return suite.compare(baseline, optimized)

    @staticmethod
    def format_markdown(comparison: BenchmarkComparison) -> str:
        return "\n".join(
            [
                "| Metric | Baseline | Optimized |",
                "|---|---:|---:|",
                f"| Throughput | {comparison.baseline.tokens_per_sec:.2f} tok/s | {comparison.optimized.tokens_per_sec:.2f} tok/s |",
                f"| GPU utilization | {comparison.baseline.gpu_utilization:.2f}% | {comparison.optimized.gpu_utilization:.2f}% |",
                f"| Memory | {comparison.baseline.memory_gb:.2f} GB | {comparison.optimized.memory_gb:.2f} GB |",
                f"| P50 latency | {comparison.baseline.p50_latency_ms:.2f} ms | {comparison.optimized.p50_latency_ms:.2f} ms |",
                f"| Speedup | 1.00x | {comparison.speedup:.2f}x |",
            ]
        )

    @staticmethod
    def write_report(comparison: BenchmarkComparison, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
        return path
