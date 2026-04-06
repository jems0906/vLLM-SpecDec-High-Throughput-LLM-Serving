from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class VendorFeatureCheck:
    name: str
    relative_path: str
    needle: str
    present: bool


class VendorVllmIntegrationProbe:
    DEFAULT_CHECKS: tuple[tuple[str, str, str], ...] = (
        (
            "speculative-cli",
            "vllm/engine/arg_utils.py",
            "--speculative-config",
        ),
        (
            "draft-model-docs",
            "docs/features/speculative_decoding/draft_model.md",
            "speculative_config",
        ),
        (
            "gpu-runner-specdecode",
            "vllm/v1/worker/gpu_model_runner.py",
            "Set up speculative decoding.",
        ),
        (
            "gpu-worker-cudagraph",
            "vllm/v1/worker/gpu_worker.py",
            "capture_model()",
        ),
        (
            "offline-example",
            "examples/offline_inference/spec_decode.py",
            "speculative_config = {",
        ),
    )

    def __init__(self, vendor_root: str | Path) -> None:
        self.vendor_root = Path(vendor_root)

    def probe(self) -> list[VendorFeatureCheck]:
        checks: list[VendorFeatureCheck] = []
        for name, relative_path, needle in self.DEFAULT_CHECKS:
            path = self.vendor_root / relative_path
            present = path.exists() and needle in path.read_text(encoding="utf-8")
            checks.append(
                VendorFeatureCheck(
                    name=name,
                    relative_path=relative_path,
                    needle=needle,
                    present=present,
                )
            )
        return checks

    def summary(self) -> dict[str, object]:
        checks = self.probe()
        checks_passed = sum(1 for check in checks if check.present)
        return {
            "vendor_root": str(self.vendor_root),
            "ready": checks_passed == len(checks),
            "checks_passed": checks_passed,
            "checks_total": len(checks),
            "checks": [asdict(check) for check in checks],
        }
