from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.config import OptimizerConfig
from vllm_specdec.vendor_integration import VendorVllmIntegrationProbe


class VendorIntegrationTests(unittest.TestCase):
    def test_vendor_probe_detects_required_upstream_features(self) -> None:
        config = OptimizerConfig()
        probe = VendorVllmIntegrationProbe(ROOT / config.vendor_vllm_dir)
        summary = probe.summary()
        self.assertTrue(summary["ready"])
        self.assertGreaterEqual(summary["checks_passed"], 4)


if __name__ == "__main__":
    unittest.main()
