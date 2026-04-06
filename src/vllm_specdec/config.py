from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OptimizerConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model_name: str = "meta-llama/Llama-3.2-1B"
    tensor_parallel_size: int = 2
    draft_tensor_parallel_size: int = 1
    max_num_seqs: int = 64
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.92
    enable_cuda_graphs: bool = True
    enforce_eager: bool = False
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 4, 8, 16, 32)
    speculative_tokens: int = 5
    served_model_name: str = "llama3-8b-specdec"
    performance_mode: str = "throughput"
    vendor_vllm_dir: Path = field(default_factory=lambda: Path("vendor") / "vllm")
    speculative_config: dict[str, Any] = field(default_factory=dict)
    compilation_config: dict[str, Any] = field(default_factory=dict)
    benchmark_baseline_cmd: str = ""
    benchmark_optimized_cmd: str = ""
    output_dir: Path = field(default_factory=lambda: Path("benchmarks") / "results")

    @classmethod
    def from_json(cls, path: str | Path) -> "OptimizerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "cuda_graph_batch_sizes" in data:
            data["cuda_graph_batch_sizes"] = tuple(data["cuda_graph_batch_sizes"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "vendor_vllm_dir" in data:
            data["vendor_vllm_dir"] = Path(data["vendor_vllm_dir"])
        return cls(**data)

    def build_speculative_config(self) -> dict[str, Any]:
        spec_config = {
            "method": "draft_model",
            "model": self.draft_model_name,
            "num_speculative_tokens": self.speculative_tokens,
            "draft_tensor_parallel_size": self.draft_tensor_parallel_size,
        }
        spec_config.update(self.speculative_config)
        return spec_config

    def build_compilation_config(self) -> dict[str, Any]:
        cudagraph_mode = "FULL_AND_PIECEWISE" if self.enable_cuda_graphs and not self.enforce_eager else "NONE"
        compilation_config: dict[str, Any] = {"cudagraph_mode": cudagraph_mode}
        if cudagraph_mode != "NONE":
            compilation_config.update(
                {
                    "cudagraph_capture_sizes": list(self.cuda_graph_batch_sizes),
                    "max_cudagraph_capture_size": max(self.cuda_graph_batch_sizes),
                }
            )
        compilation_config.update(self.compilation_config)
        return compilation_config

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "draft_model_name": self.draft_model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "draft_tensor_parallel_size": self.draft_tensor_parallel_size,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "enforce_eager": self.enforce_eager,
            "cuda_graph_batch_sizes": list(self.cuda_graph_batch_sizes),
            "speculative_tokens": self.speculative_tokens,
            "served_model_name": self.served_model_name,
            "performance_mode": self.performance_mode,
            "vendor_vllm_dir": str(self.vendor_vllm_dir),
            "speculative_config": self.build_speculative_config(),
            "compilation_config": self.build_compilation_config(),
            "benchmark_baseline_cmd": self.benchmark_baseline_cmd,
            "benchmark_optimized_cmd": self.benchmark_optimized_cmd,
            "output_dir": str(self.output_dir),
        }

    def validate(self) -> None:
        if "Llama-3" not in self.model_name and "Llama-3" not in self.draft_model_name:
            raise ValueError("This project template is tuned for Llama-3 family models.")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.draft_tensor_parallel_size not in (1, self.tensor_parallel_size):
            raise ValueError("draft_tensor_parallel_size must be 1 or equal to tensor_parallel_size")
        if self.speculative_tokens < 1:
            raise ValueError("speculative_tokens must be >= 1")
        if self.performance_mode not in {"balanced", "interactivity", "throughput"}:
            raise ValueError("performance_mode must be balanced, interactivity, or throughput")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
