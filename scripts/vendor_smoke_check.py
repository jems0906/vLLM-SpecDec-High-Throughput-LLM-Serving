from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.config import OptimizerConfig
from vllm_specdec.engine import VLLMSpecDecEngine
from vllm_specdec.vendor_integration import VendorVllmIntegrationProbe


def main() -> int:
    config = OptimizerConfig.from_json(ROOT / "configs" / "llama3_8b_specdec.json")
    vendor_root = config.vendor_vllm_dir
    if not vendor_root.is_absolute():
        vendor_root = (ROOT / vendor_root).resolve()

    summary = VendorVllmIntegrationProbe(vendor_root).summary()
    engine = VLLMSpecDecEngine(config)
    tp_plan = engine.tp_launcher.build_plan(
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
    )

    print("=== Vendor vLLM Integration Probe ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Recommended Optimized Launch Command ===")
    print(tp_plan.shell_command())
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
