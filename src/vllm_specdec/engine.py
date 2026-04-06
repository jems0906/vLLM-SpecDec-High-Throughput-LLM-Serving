from __future__ import annotations

import json
from dataclasses import asdict
from typing import cast

from .config import OptimizerConfig
from .cuda_graphs import CUDAGraphCaptureManager
from .tensor_parallel import TensorParallelLauncher


class VLLMSpecDecEngine:
    def __init__(self, config: OptimizerConfig) -> None:
        config.validate()
        self.config = config
        self.capture_manager = CUDAGraphCaptureManager(
            enabled=config.enable_cuda_graphs,
            capture_batch_sizes=config.cuda_graph_batch_sizes,
        )
        self.tp_launcher = TensorParallelLauncher(world_size=config.tensor_parallel_size)

    @staticmethod
    def _compact_json(payload: dict[str, object]) -> str:
        return json.dumps(payload, separators=(",", ":"))

    def build_server_command(self, host: str = "0.0.0.0", port: int = 8000) -> list[str]:
        speculative_config = self.config.build_speculative_config()
        compilation_config = self.config.build_compilation_config()
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            self.config.model_name,
            "--served-model-name",
            self.config.served_model_name,
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--max-num-seqs",
            str(self.config.max_num_seqs),
            "--max-model-len",
            str(self.config.max_model_len),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--performance-mode",
            self.config.performance_mode,
            "--enable-prefix-caching",
            "--disable-log-stats",
            "--speculative-config",
            self._compact_json(speculative_config),
            "--compilation-config",
            self._compact_json(compilation_config),
        ]
        if self.config.enforce_eager:
            command.append("--enforce-eager")
        return command

    def build_runtime_plan(self) -> dict[str, object]:
        speculative_config = self.config.build_speculative_config()
        compilation_config = self.config.build_compilation_config()
        prefill = [
            asdict(self.capture_manager.plan_capture("prefill", batch_size, min(self.config.max_model_len, 1024)))
            for batch_size in self.config.cuda_graph_batch_sizes
        ]
        decode = [
            asdict(self.capture_manager.plan_capture("decode", batch_size, 1))
            for batch_size in self.config.cuda_graph_batch_sizes
        ]
        tp_plan = self.tp_launcher.build_plan(
            model_name=self.config.model_name,
            draft_model_name=self.config.draft_model_name,
            num_speculative_tokens=self.config.speculative_tokens,
            draft_tensor_parallel_size=self.config.draft_tensor_parallel_size,
            speculative_config=speculative_config,
            compilation_config=compilation_config,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            served_model_name=self.config.served_model_name,
            performance_mode=self.config.performance_mode,
        )
        return {
            "server_command": self.build_server_command(),
            "speculative_config": speculative_config,
            "compilation_config": compilation_config,
            "prefill_capture_plan": prefill,
            "decode_capture_plan": decode,
            "tensor_parallel": {
                "world_size": tp_plan.world_size,
                "devices": tp_plan.devices,
                "layer_slices": tp_plan.layer_slices,
                "env": tp_plan.env,
                "command": tp_plan.command,
            },
        }

    def describe(self) -> str:
        plan = self.build_runtime_plan()
        prefill_plan = cast(list[dict[str, object]], plan["prefill_capture_plan"])
        decode_plan = cast(list[dict[str, object]], plan["decode_capture_plan"])
        return (
            f"Model: {self.config.model_name}\n"
            f"Draft model: {self.config.draft_model_name}\n"
            f"Tensor parallel size: {self.config.tensor_parallel_size}\n"
            f"Performance mode: {self.config.performance_mode}\n"
            f"CUDA Graphs enabled: {self.config.enable_cuda_graphs}\n"
            f"Prefill capture candidates: {len(prefill_plan)}\n"
            f"Decode capture candidates: {len(decode_plan)}"
        )
