from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_specdec.config import OptimizerConfig
from vllm_specdec.engine import VLLMSpecDecEngine


def _runtime_prereq_errors(env: dict[str, str]) -> list[str]:
    errors: list[str] = []
    has_torch = importlib.util.find_spec("torch") is not None
    search_path = env.get("PYTHONPATH", "")
    has_vllm = importlib.util.find_spec("vllm") is not None or "vendor\\vllm" in search_path or "vendor/vllm" in search_path
    has_torchrun = shutil.which("torchrun") is not None

    if not has_torch:
        errors.append("PyTorch is not installed in the selected environment.")
    if not has_vllm:
        errors.append("vLLM is not importable; install it or use the cloned `vendor/vllm` path in a Linux/CUDA environment.")
    if not has_torchrun and has_torch:
        errors.append("`torchrun` is not on PATH; install a full PyTorch runtime or invoke from a Linux/CUDA shell.")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch or preview the 2x A100 tensor-parallel vLLM server.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "llama3_8b_specdec.json"))
    parser.add_argument("--model")
    parser.add_argument("--draft-model")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--num-speculative-tokens", type=int)
    parser.add_argument("--use-system-vllm", action="store_true", help="Do not prepend `vendor/vllm` to PYTHONPATH.")
    parser.add_argument("--show-runtime-plan", action="store_true", help="Print the full JSON runtime plan in addition to the launch command.")
    parser.add_argument("--execute", action="store_true", help="Actually launch torchrun instead of printing the plan.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OptimizerConfig.from_json(args.config)

    if args.model:
        config.model_name = args.model
    if args.draft_model:
        config.draft_model_name = args.draft_model
    if args.tensor_parallel_size:
        config.tensor_parallel_size = args.tensor_parallel_size
    if args.num_speculative_tokens:
        config.speculative_tokens = args.num_speculative_tokens

    engine = VLLMSpecDecEngine(config)
    runtime_plan = engine.build_runtime_plan()
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

    env = os.environ.copy()
    env.update(tp_plan.env)

    vendor_vllm_dir = config.vendor_vllm_dir
    if not vendor_vllm_dir.is_absolute():
        vendor_vllm_dir = (ROOT / vendor_vllm_dir).resolve()
    if not args.use_system_vllm and vendor_vllm_dir.exists():
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(vendor_vllm_dir) if not current_pythonpath else f"{vendor_vllm_dir}{os.pathsep}{current_pythonpath}"

    print("=== Tensor Parallel Environment ===")
    for key, value in tp_plan.env.items():
        print(f"{key}={value}")
    if "PYTHONPATH" in env:
        print(f"PYTHONPATH={env['PYTHONPATH']}")

    print("\n=== Layer Placement ===")
    for rank, layer_slice in enumerate(tp_plan.layer_slices):
        print(f"rank{rank}: layers[{layer_slice[0]}:{layer_slice[1]}]")

    print("\n=== Launch Command ===")
    print(tp_plan.shell_command())

    if args.show_runtime_plan:
        print("\n=== Runtime Plan (JSON) ===")
        print(json.dumps(runtime_plan, indent=2))

    if not args.execute:
        return 0

    prereq_errors = _runtime_prereq_errors(env)
    if prereq_errors:
        print("\n=== Runtime Preflight Failed ===")
        for error in prereq_errors:
            print(f"- {error}")
        print("\nUse Docker or a Linux host with 2x A100 GPUs, CUDA, PyTorch, and vLLM installed.")
        return 1

    completed = subprocess.run(tp_plan.command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
