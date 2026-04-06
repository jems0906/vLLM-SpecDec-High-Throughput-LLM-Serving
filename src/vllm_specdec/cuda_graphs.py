from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


@dataclass(slots=True)
class GraphCaptureRecord:
    phase: str
    batch_size: int
    sequence_length: int
    captured: bool
    reason: str


class CUDAGraphCaptureManager:
    def __init__(self, enabled: bool = True, capture_batch_sizes: tuple[int, ...] = (1, 4, 8, 16, 32)) -> None:
        self.enabled = enabled
        self.capture_batch_sizes = set(capture_batch_sizes)
        self.records: list[GraphCaptureRecord] = []

    def plan_capture(self, phase: str, batch_size: int, sequence_length: int) -> GraphCaptureRecord:
        if not self.enabled:
            record = GraphCaptureRecord(phase, batch_size, sequence_length, False, "cuda graphs disabled")
        elif batch_size not in self.capture_batch_sizes:
            record = GraphCaptureRecord(phase, batch_size, sequence_length, False, "batch size not in capture set")
        elif torch is None or not torch.cuda.is_available():
            record = GraphCaptureRecord(phase, batch_size, sequence_length, False, "CUDA unavailable in current environment")
        else:
            record = GraphCaptureRecord(phase, batch_size, sequence_length, True, "eligible for graph capture")
        self.records.append(record)
        return record

    def maybe_capture(
        self,
        phase: str,
        batch_size: int,
        sequence_length: int,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, GraphCaptureRecord]:
        record = self.plan_capture(phase, batch_size, sequence_length)
        if not record.captured or torch is None:
            return fn(*args, **kwargs), record

        warmup_output = fn(*args, **kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = fn(*args, **kwargs)
        return {"graph": graph, "warmup_output": warmup_output, "captured_output": captured_output}, record

    def summary(self) -> dict[str, int]:
        captured = sum(1 for record in self.records if record.captured)
        skipped = len(self.records) - captured
        return {"captured": captured, "skipped": skipped}
