from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.spec_decode import SpeculativeDecoder, accept_tokens


class SpeculativeDecodeTests(unittest.TestCase):
    def test_accept_tokens_stops_on_first_mismatch(self) -> None:
        window = accept_tokens([10, 11, 12, 13], [10, 11, 77, 88])
        self.assertEqual(window.accepted_tokens, [10, 11])
        self.assertEqual(window.rejected_token, 77)
        self.assertAlmostEqual(window.acceptance_rate, 0.5)

    def test_merge_step_appends_rejected_target_token(self) -> None:
        decoder = SpeculativeDecoder(num_speculative_tokens=4)
        merged = decoder.merge_step([1, 2, 3, 4], [1, 2, 9, 10])
        self.assertEqual(merged, [1, 2, 9])


if __name__ == "__main__":
    unittest.main()
