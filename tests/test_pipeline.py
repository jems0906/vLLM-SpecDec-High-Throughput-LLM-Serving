from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.reporting import ProjectRunSummary


class PipelineSummaryTests(unittest.TestCase):
    def test_markdown_summary_contains_project_targets(self) -> None:
        summary = ProjectRunSummary.synthetic(ROOT / "vendor" / "vllm")
        markdown = summary.to_markdown()
        self.assertIn("2.49x", markdown)
        self.assertIn("kernel-launch-overhead", markdown)
        self.assertIn("vendor/vllm", markdown.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
