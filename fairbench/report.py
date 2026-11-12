"""Generate HTML reports for benchmark runs."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FairBench Report — {model_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 0; }}
        .model-info {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .info-item {{ padding: 10px; background: #ecf0f1; border-radius: 4px; }}
        .info-label {{ font-size: 0.85em; color: #7f8c8d; text-transform: uppercase; }}
        .info-value {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
        .chart-container {{ position: relative; height: 400px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .stat-card {{ text-align: center; padding: 15px; background: #ecf0f1; border-radius: 8px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 0.9em; color: #7f8c8d; }}
        .swebench-table {{ width: 100%; border-collapse: collapse; }}
        .swebench-table th, .swebench-table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .swebench-table th {{ background: #ecf0f1; font-weight: 600; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .error {{ color: #f39c12; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FairBench Report</h1>

        <div class="card">
            <h2>Model Information</h2>
            <div class="model-info">
                <div class="info-item">
                    <div class="info-label">Model</div>
                    <div class="info-value">{model_name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Quantization</div>
                    <div class="info-value">{quantization}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Context</div>
                    <div class="info-value">{n_ctx}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">GPU Layers</div>
                    <div class="info-value">{n_gpu_layers}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Threads</div>
                    <div class="info-value">{n_threads}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Timestamp</div>
                    <div class="info-value">{timestamp}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Quality Benchmarks</h2>
            <div class="chart-container">
                <canvas id="qualityChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Speed & Memory</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{throughput}</div>
                    <div class="stat-label">Tokens/sec</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{ttft}</div>
                    <div class="stat-label">TTFT (ms)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{tpot}</div>
                    <div class="stat-label">TPOT (ms)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{vram}</div>
                    <div class="stat-label">VRAM (MB)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{ram}</div>
                    <div class="stat-label">RAM (MB)</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>SWE-bench Results</h2>
            <table class="swebench-table">
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td>Resolved</td><td class="pass">{swebench_resolved}/{swebench_total}</td></tr>
                    <tr><td>Failed</td><td class="fail">{swebench_failed}</td></tr>
                    <tr><td>Errors</td><td class="error">{swebench_errors}</td></tr>
                    <tr><td>Score</td><td>{swebench_score}%</td></tr>
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Comparison</h2>
            <p>View this model on the <a href="https://fairbench.nousresearch.com/leaderboard" target="_blank">FairBench Leaderboard</a>.</p>
        </div>
    </div>

    <script>
        const labels = {labels};
        const scores = {scores};
        const ctx = document.getElementById('qualityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Score',
                    data: scores,
                    backgroundColor: 'rgba(52, 152, 219, 0.6)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, max: 1.0 }}
                }}
            }}
        }});
    </script>
</body>
</html>"""


class ReportEngine:
    """Generates HTML reports for benchmark runs."""

    def __init__(self, results_dir: str | Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else Path.home() / ".fairbench" / "results" / "run"
        self.reports_dir = Path.home() / ".fairbench" / "reports"

    def generate_html(self, run_id: str) -> str:
        """Generate HTML report for a specific run."""
        run_file = self.results_dir / f"{run_id}.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run file not found: {run_file}")

        with open(run_file, "r") as f:
            data = json.load(f)

        # Extract quality benchmarks for chart
        quality_benchmarks = data.get("quality_benchmarks", {})
        labels = list(quality_benchmarks.keys())
        scores = [v["score"] if isinstance(v, dict) else v for v in quality_benchmarks.values()]

        # Extract speed/memory stats
        speed = data.get("speed", {})
        memory = data.get("memory", {})
        swebench = data.get("swebench", {})

        html = HTML_TEMPLATE.format(
            model_name=data.get("model_name", "unknown"),
            quantization=data.get("quantization", "unknown"),
            n_ctx=data.get("n_ctx", ""),
            n_gpu_layers=data.get("n_gpu_layers", ""),
            n_threads=data.get("n_threads", ""),
            timestamp=data.get("timestamp", ""),
            throughput=speed.get("throughput_tokens_per_sec", 0),
            ttft=speed.get("ttft_ms", 0),
            tpot=speed.get("tpot_ms", 0),
            vram=memory.get("vram_peak_mb", 0),
            ram=memory.get("ram_mb", 0),
            swebench_resolved=swebench.get("resolved", 0),
            swebench_total=swebench.get("total", 0),
            swebench_failed=swebench.get("failed", 0),
            swebench_errors=swebench.get("errors", 0),
            swebench_score=swebench.get("score", 0),
            labels=json.dumps(labels),
            scores=json.dumps(scores),
        )

        return html

    def save_report(self, run_id: str, open_browser: bool = False) -> Path:
        """Save HTML report and optionally open in browser."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.reports_dir / f"{run_id}_report.html"

        html = self.generate_html(run_id)
        report_path.write_text(html, encoding="utf-8")

        if open_browser:
            webbrowser.open(f"file://{report_path.absolute()}")

        return report_path
