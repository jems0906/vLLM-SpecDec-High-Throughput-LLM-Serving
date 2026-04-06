from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.live_benchmark import ServingBenchmarkSummary, compare_summaries


class LiveComparisonTests(unittest.TestCase):
    def test_compare_summaries_reports_speedup_and_targets(self) -> None:
        baseline = ServingBenchmarkSummary(
            label="baseline-vllm",
            tokens_per_sec=45.0,
            gpu_utilization=68.0,
            memory_gb=16.50,
            p50_latency_ms=87.0,
            completed_requests=32,
            total_tokens=1440,
        )
        optimized = ServingBenchmarkSummary(
            label="optimized-specdec",
            tokens_per_sec=112.0,
            gpu_utilization=92.0,
            memory_gb=13.53,
            p50_latency_ms=34.0,
            completed_requests=32,
            total_tokens=3584,
        )

        comparison = compare_summaries(baseline, optimized)
        self.assertAlmostEqual(comparison.speedup, 112.0 / 45.0, places=2)
        self.assertTrue(comparison.meets_throughput_target)


if __name__ == "__main__":
    unittest.main()
