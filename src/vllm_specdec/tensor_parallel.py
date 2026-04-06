from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(slots=True)
class TensorParallelPlan:
    world_size: int
    devices: list[int]
    layer_slices: list[tuple[int, int]]
    env: dict[str, str]
    command: list[str]

    @staticmethod
    def _quote_for_powershell(arg: str) -> str:
        if not arg:
            return "''"
        if arg.startswith("{") or arg.startswith("[") or any(ch in arg for ch in (' ', '"')):
            return "'" + arg.replace("'", "''") + "'"
        return arg

    def shell_command(self) -> str:
        return " ".join(self._quote_for_powershell(arg) for arg in self.command)


class TensorParallelLauncher:
    def __init__(self, world_size: int = 2, total_layers: int = 32) -> None:
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        self.world_size = world_size
        self.total_layers = total_layers

    @staticmethod
    def _compact_json(payload: dict[str, object]) -> str:
        return json.dumps(payload, separators=(",", ":"))

    def partition_layers(self) -> list[tuple[int, int]]:
        base = self.total_layers // self.world_size
        remainder = self.total_layers % self.world_size
        partitions: list[tuple[int, int]] = []
        start = 0
        for rank in range(self.world_size):
            width = base + (1 if rank < remainder else 0)
            end = start + width
            partitions.append((start, end))
            start = end
        return partitions

    def build_plan(
        self,
        model_name: str,
        draft_model_name: str,
        num_speculative_tokens: int = 5,
        draft_tensor_parallel_size: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
        master_port: int = 29500,
        speculative_config: dict[str, object] | None = None,
        compilation_config: dict[str, object] | None = None,
        max_num_seqs: int = 64,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.92,
        served_model_name: str = "llama3-8b-specdec",
        performance_mode: str = "throughput",
    ) -> TensorParallelPlan:
        devices = list(range(self.world_size))
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in devices),
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
        }
        if speculative_config is None:
            speculative_config = {
                "method": "draft_model",
                "model": draft_model_name,
                "num_speculative_tokens": num_speculative_tokens,
                "draft_tensor_parallel_size": draft_tensor_parallel_size,
            }
        if compilation_config is None:
            compilation_config = {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [1, 4, 8, 16, 32],
                "max_cudagraph_capture_size": 32,
            }
        command = [
            "torchrun",
            "--nproc_per_node",
            str(self.world_size),
            "--master_port",
            str(master_port),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            model_name,
            "--served-model-name",
            served_model_name,
            "--tensor-parallel-size",
            str(self.world_size),
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--performance-mode",
            performance_mode,
            "--enable-prefix-caching",
            "--disable-log-stats",
            "--speculative-config",
            self._compact_json(speculative_config),
            "--compilation-config",
            self._compact_json(compilation_config),
        ]
        return TensorParallelPlan(
            world_size=self.world_size,
            devices=devices,
            layer_slices=self.partition_layers(),
            env=env,
            command=command,
        )
