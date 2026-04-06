from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.benchmark import BenchmarkSuite


class BenchmarkTests(unittest.TestCase):
    def test_synthetic_target_matches_project_goal(self) -> None:
        comparison = BenchmarkSuite.synthetic_target()
        self.assertEqual(comparison.baseline.tokens_per_sec, 45.0)
        self.assertEqual(comparison.optimized.tokens_per_sec, 112.0)
        self.assertGreaterEqual(comparison.speedup, 2.48)
        self.assertTrue(comparison.meets_throughput_target)


if __name__ == "__main__":
    unittest.main()
