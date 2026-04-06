from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import BenchmarkComparison, BenchmarkSuite
from .nsys_profiler import BottleneckFinding, NsightAnalyzer
from .vendor_integration import VendorVllmIntegrationProbe


@dataclass(slots=True)
class ProjectRunSummary:
    vendor_summary: dict[str, object]
    benchmark: BenchmarkComparison
    findings: list[BottleneckFinding]

    @classmethod
    def synthetic(cls, vendor_root: str | Path) -> "ProjectRunSummary":
        analyzer = NsightAnalyzer()
        sample_trace = """
        cudaLaunchKernel 18.2%
        Memcpy HtoD 11.5%
        GPU utilization 72.0%
        pthread_cond_wait 7.3%
        """
        return cls(
            vendor_summary=VendorVllmIntegrationProbe(vendor_root).summary(),
            benchmark=BenchmarkSuite.synthetic_target(),
            findings=analyzer.analyze_text(sample_trace),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "vendor_summary": self.vendor_summary,
            "benchmark": self.benchmark.to_dict(),
            "findings": [asdict(finding) for finding in self.findings],
        }

    def to_markdown(self) -> str:
        findings_block = "\n".join(
            f"- **{finding.category}** ({finding.severity}): {finding.evidence}"
            for finding in self.findings
        )
        return "\n".join(
            [
                "# Project 1 Run Summary",
                "",
                f"- Vendor root: `{self.vendor_summary['vendor_root']}`",
                f"- Upstream ready: `{self.vendor_summary['ready']}` ({self.vendor_summary['checks_passed']}/{self.vendor_summary['checks_total']} checks)",
                f"- Throughput: `{self.benchmark.baseline.tokens_per_sec:.0f} tok/s -> {self.benchmark.optimized.tokens_per_sec:.0f} tok/s`",
                f"- Speedup: `{self.benchmark.speedup:.2f}x`",
                f"- GPU utilization: `{self.benchmark.baseline.gpu_utilization:.0f}% -> {self.benchmark.optimized.gpu_utilization:.0f}%`",
                f"- Memory reduction: `{self.benchmark.memory_reduction_pct:.1f}%`",
                "",
                "## Nsight findings",
                findings_block,
            ]
        )

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path = target_dir / "project1_summary.json"
        md_path = target_dir / "project1_summary.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path
