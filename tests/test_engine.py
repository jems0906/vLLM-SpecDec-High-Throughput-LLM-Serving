from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.config import OptimizerConfig
from vllm_specdec.engine import VLLMSpecDecEngine


class EnginePlanTests(unittest.TestCase):
    def test_runtime_plan_includes_specdec_and_tp(self) -> None:
        config = OptimizerConfig()
        engine = VLLMSpecDecEngine(config)
        plan = engine.build_runtime_plan()
        server_command = " ".join(plan["server_command"])
        self.assertIn("--speculative-config", server_command)
        self.assertIn("--compilation-config", server_command)
        self.assertIn("--tensor-parallel-size", server_command)
        self.assertEqual(plan["tensor_parallel"]["layer_slices"], [(0, 16), (16, 32)])

        shell_command = engine.tp_launcher.build_plan(
            model_name=config.model_name,
            draft_model_name=config.draft_model_name,
            num_speculative_tokens=config.speculative_tokens,
            draft_tensor_parallel_size=config.draft_tensor_parallel_size,
            speculative_config=config.build_speculative_config(),
            compilation_config=config.build_compilation_config(),
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            served_model_name=config.served_model_name,
            performance_mode=config.performance_mode,
        ).shell_command()
        self.assertIn("--speculative-config '{\"method\":\"draft_model\"", shell_command)


if __name__ == "__main__":
    unittest.main()
