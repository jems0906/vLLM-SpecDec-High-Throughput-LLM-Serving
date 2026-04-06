from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BottleneckFinding:
    category: str
    severity: str
    evidence: str
    recommendation: str


class NsightAnalyzer:
    KERNEL_LAUNCH_RE = re.compile(r"cudaLaunchKernel.*?([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
    MEMCPY_RE = re.compile(r"(?:cudaMemcpy|Memcpy HtoD|Memcpy DtoH).*?([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
    GPU_UTIL_RE = re.compile(r"(?:gpu\s+utilization|sm\s+active).*?([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
    CPU_STALL_RE = re.compile(r"(?:pthread_cond_wait|sched_yield).*?([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)

    @staticmethod
    def _match_value(pattern: re.Pattern[str], text: str) -> float | None:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    def analyze_text(self, text: str) -> list[BottleneckFinding]:
        findings: list[BottleneckFinding] = []

        kernel_launch = self._match_value(self.KERNEL_LAUNCH_RE, text)
        if kernel_launch is not None and kernel_launch > 12:
            findings.append(
                BottleneckFinding(
                    category="kernel-launch-overhead",
                    severity="high",
                    evidence=f"cudaLaunchKernel consumes {kernel_launch:.1f}% of sampled time",
                    recommendation="Capture graph-safe prefill/decode paths with CUDA Graphs to reduce CPU dispatch overhead.",
                )
            )

        memcpy = self._match_value(self.MEMCPY_RE, text)
        if memcpy is not None and memcpy > 10:
            findings.append(
                BottleneckFinding(
                    category="memory-transfer",
                    severity="medium",
                    evidence=f"Memcpy activity consumes {memcpy:.1f}% of sampled time",
                    recommendation="Keep KV cache and sampling buffers resident on-device and batch host-to-device transfers.",
                )
            )

        gpu_util = self._match_value(self.GPU_UTIL_RE, text)
        if gpu_util is not None and gpu_util < 85:
            findings.append(
                BottleneckFinding(
                    category="gpu-underutilization",
                    severity="high",
                    evidence=f"GPU utilization is only {gpu_util:.1f}%",
                    recommendation="Increase batch occupancy, enable tensor parallelism, and reduce decode launch gaps.",
                )
            )

        cpu_stall = self._match_value(self.CPU_STALL_RE, text)
        if cpu_stall is not None and cpu_stall > 5:
            findings.append(
                BottleneckFinding(
                    category="cpu-synchronization",
                    severity="medium",
                    evidence=f"CPU synchronization stalls account for {cpu_stall:.1f}%",
                    recommendation="Avoid unnecessary host synchronization and overlap token scheduling with GPU execution.",
                )
            )

        if not findings:
            findings.append(
                BottleneckFinding(
                    category="healthy-trace",
                    severity="info",
                    evidence="No dominant bottleneck crossed the configured thresholds.",
                    recommendation="Keep current CUDA Graph and tensor-parallel settings; validate again under heavier batch load.",
                )
            )
        return findings

    def analyze_file(self, path: str | Path) -> list[BottleneckFinding]:
        text = Path(path).read_text(encoding="utf-8")
        return self.analyze_text(text)
