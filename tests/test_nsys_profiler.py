from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.nsys_profiler import NsightAnalyzer


class NsightAnalyzerTests(unittest.TestCase):
    def test_detects_kernel_launch_and_gpu_utilization_issues(self) -> None:
        summary = """
        cudaLaunchKernel 18.2%
        GPU utilization 72.0%
        pthread_cond_wait 7.3%
        """
        findings = NsightAnalyzer().analyze_text(summary)
        categories = {finding.category for finding in findings}
        self.assertIn("kernel-launch-overhead", categories)
        self.assertIn("gpu-underutilization", categories)
        self.assertIn("cpu-synchronization", categories)


if __name__ == "__main__":
    unittest.main()
