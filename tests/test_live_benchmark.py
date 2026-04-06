from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.live_benchmark import aggregate_results


class LiveBenchmarkTests(unittest.TestCase):
    def test_aggregate_results_computes_tokens_and_latency(self) -> None:
        summary = aggregate_results(
            durations=[1.0, 2.0, 1.5],
            completion_tokens=[40, 50, 45],
            gpu_utilization=92.0,
            memory_gb=13.53,
            label="optimized-specdec",
        )
        self.assertEqual(summary.label, "optimized-specdec")
        self.assertAlmostEqual(summary.tokens_per_sec, 67.5, places=2)
        self.assertAlmostEqual(summary.p50_latency_ms, 1500.0, places=2)


if __name__ == "__main__":
    unittest.main()
