from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .benchmark import BenchmarkComparison, BenchmarkResult, BenchmarkSuite


@dataclass(slots=True)
class ServingBenchmarkSummary:
    label: str
    tokens_per_sec: float
    gpu_utilization: float
    memory_gb: float
    p50_latency_ms: float
    completed_requests: int
    total_tokens: int

    def to_console_lines(self) -> list[str]:
        return [
            f"label: {self.label}",
            f"tokens/sec: {self.tokens_per_sec:.2f}",
            f"GPU utilization: {self.gpu_utilization:.2f}%",
            f"memory: {self.memory_gb:.2f} GB",
            f"P50 latency: {self.p50_latency_ms:.2f} ms",
            f"completed_requests: {self.completed_requests}",
            f"total_tokens: {self.total_tokens}",
        ]

    def to_benchmark_result(self) -> BenchmarkResult:
        return BenchmarkResult(
            label=self.label,
            tokens_per_sec=self.tokens_per_sec,
            gpu_utilization=self.gpu_utilization,
            memory_gb=self.memory_gb,
            p50_latency_ms=self.p50_latency_ms,
            notes=f"completed_requests={self.completed_requests}, total_tokens={self.total_tokens}",
        )


def load_prompts(path: str | Path) -> list[str]:
    prompts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            prompts.append(str(data.get("prompt", "")))
        else:
            prompts.append(str(data))
    return prompts


def aggregate_results(
    durations: list[float],
    completion_tokens: list[int],
    gpu_utilization: float,
    memory_gb: float,
    label: str,
) -> ServingBenchmarkSummary:
    wall_clock_duration = max(durations) if durations else 0.0
    total_tokens = sum(completion_tokens)
    p50_latency_ms = statistics.median(durations) * 1000 if durations else 0.0
    tokens_per_sec = total_tokens / wall_clock_duration if wall_clock_duration else 0.0
    return ServingBenchmarkSummary(
        label=label,
        tokens_per_sec=tokens_per_sec,
        gpu_utilization=gpu_utilization,
        memory_gb=memory_gb,
        p50_latency_ms=p50_latency_ms,
        completed_requests=len(durations),
        total_tokens=total_tokens,
    )


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, ready_timeout: float = 300.0, poll_interval: float = 2.0) -> None:
    health_urls = [base_url.rstrip("/") + "/health", base_url.rstrip("/") + "/v1/models"]
    deadline = time.monotonic() + ready_timeout
    last_error: str | None = None

    while time.monotonic() < deadline:
        for url in health_urls:
            try:
                with request.urlopen(url, timeout=5.0) as response:
                    if response.status < 500:
                        return
            except (error.URLError, TimeoutError, ValueError) as exc:
                last_error = str(exc)
        time.sleep(poll_interval)

    raise TimeoutError(f"Server at {base_url} did not become ready within {ready_timeout}s. Last error: {last_error}")


def _run_one_request(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[float, int]:
    endpoint = base_url.rstrip("/") + "/v1/completions"
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    start = time.perf_counter()
    result = _post_json(endpoint, payload, timeout)
    duration = time.perf_counter() - start
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    completion_tokens = int(usage.get("completion_tokens", max_tokens))
    return duration, completion_tokens


def run_live_benchmark(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    timeout: float,
    gpu_utilization: float,
    memory_gb: float,
    label: str,
    ready_timeout: float = 300.0,
) -> ServingBenchmarkSummary:
    wait_for_server(base_url, ready_timeout=ready_timeout)
    durations: list[float] = []
    token_counts: list[int] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(_run_one_request, base_url, model, prompt, max_tokens, timeout)
            for prompt in prompts
        ]
        for future in futures:
            duration, tokens = future.result()
            durations.append(duration)
            token_counts.append(tokens)
    return aggregate_results(durations, token_counts, gpu_utilization, memory_gb, label)


def compare_summaries(
    baseline: ServingBenchmarkSummary,
    optimized: ServingBenchmarkSummary,
    throughput_target: float = 2.5,
) -> BenchmarkComparison:
    suite = BenchmarkSuite(throughput_target=throughput_target)
    return suite.compare(baseline.to_benchmark_result(), optimized.to_benchmark_result())
