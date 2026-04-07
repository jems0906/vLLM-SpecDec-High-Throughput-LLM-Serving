from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGET_PROFILE = ROOT / "benchmarks" / "target_a100_report.json"


def load_target_profile() -> dict[str, Any]:
    try:
        raw = json.loads(TARGET_PROFILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {"error": "target profile unavailable"}
    except Exception:
        return {"error": "target profile unavailable"}


class RenderStatusHandler(BaseHTTPRequestHandler):
    server_version = "vLLMSpecDecRender/1.0"

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str, status: int = 200) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        profile = load_target_profile()

        if self.path in ("/health", "/healthz"):
            self._send_json(
                {
                    "status": "ok",
                    "service": "vllm-specdec-render-demo",
                    "mode": "cpu-safe-demo",
                    "gpu_required_for_live_inference": True,
                }
            )
            return

        if self.path in ("/summary", "/api/summary"):
            self._send_json(
                {
                    "project": "vLLM-SpecDec",
                    "deployment": "Render demo",
                    "note": "This Render deployment exposes a project summary endpoint. Full live vLLM inference requires a Linux/NVIDIA GPU host.",
                    "target_profile": profile,
                }
            )
            return

        if self.path in ("/", "/index.html"):
            comparison = profile.get("comparison", {}) if isinstance(profile, dict) else {}
            baseline = profile.get("baseline", {}) if isinstance(profile, dict) else {}
            optimized = profile.get("optimized", {}) if isinstance(profile, dict) else {}
            html = f"""
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>vLLM-SpecDec Render Demo</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }}
      code {{ background: #f4f4f4; padding: 0.15rem 0.35rem; border-radius: 4px; }}
      .card {{ max-width: 900px; padding: 1.2rem 1.4rem; border: 1px solid #ddd; border-radius: 12px; }}
      ul {{ padding-left: 1.2rem; }}
    </style>
  </head>
  <body>
    <div class=\"card\">
      <h1>vLLM-SpecDec Render Demo</h1>
      <p>This is a <strong>CPU-safe demo/status deployment</strong> for the project.</p>
      <p>The full speculative decoding service with <code>vLLM</code> requires a <strong>Linux/NVIDIA GPU runtime</strong> and is not executed on Render in this demo mode.</p>
      <h2>Target A100 Profile</h2>
      <ul>
        <li>Baseline throughput: <strong>{baseline.get('tokens_per_sec', 'n/a')} tok/s</strong></li>
        <li>Optimized throughput: <strong>{optimized.get('tokens_per_sec', 'n/a')} tok/s</strong></li>
        <li>Speedup target: <strong>{comparison.get('speedup', 'n/a')}x</strong></li>
        <li>GPU utilization target: <strong>{optimized.get('gpu_utilization', 'n/a')}%</strong></li>
        <li>Memory reduction target: <strong>{comparison.get('memory_reduction_pct', 'n/a')}%</strong></li>
      </ul>
      <p>Endpoints: <a href=\"/health\">/health</a> and <a href=\"/summary\">/summary</a></p>
    </div>
  </body>
</html>
"""
            self._send_html(html)
            return

        self._send_json({"error": "not found"}, status=404)


def main() -> None:
    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), RenderStatusHandler)
    print(f"Render demo server listening on 0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
