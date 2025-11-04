from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from telemetry.metrics import TelemetryMetrics, TelemetrySummary

matplotlib.use("Agg")


def generate_report(
    telemetry_path: str | Path = "runs/telemetry.csv",
    output_dir: str | Path = "reports/last_run",
) -> Path:
    telemetry_file = Path(telemetry_path)
    if not telemetry_file.exists():
        message = f"Telemetry file not found: {telemetry_file}"
        raise FileNotFoundError(message)
    metrics = TelemetryMetrics.from_csv(telemetry_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_data = metrics.to_dict()
    (out_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))
    _render_charts(metrics, out_dir)
    summary = _render_markdown(metrics_data)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary)
    return summary_path


def _render_charts(metrics: TelemetryMetrics, out_dir: Path) -> None:
    conversion = metrics.conversion_rate_per_mechanic()
    if conversion:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(conversion.keys(), conversion.values(), color="#2E86DE")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Conversion Rate")
        ax.set_title("Mechanic Conversion")
        fig.tight_layout()
        fig.savefig(out_dir / "conversion.png")
        plt.close(fig)
    boost = metrics.average_boost_per_attempt()
    if boost:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(boost.keys(), boost.values(), color="#58B19F")
        ax.set_ylabel("Average Boost")
        ax.set_title("Boost Usage")
        fig.tight_layout()
        fig.savefig(out_dir / "boost.png")
        plt.close(fig)


def _render_markdown(metrics: TelemetrySummary) -> str:
    lines = ["# RLBot Pro Telemetry Report", ""]
    lines.append(f"Total rows: {metrics['rows']}")
    lines.append("")
    lines.append("## Conversion Rate")
    for mechanic, rate in metrics["conversion_rate"].items():
        lines.append(f"- {mechanic}: {rate:.2f}")
    lines.append("")
    lines.append("## Average Boost")
    for mechanic, boost_value in metrics["average_boost_per_attempt"].items():
        lines.append(f"- {mechanic}: {boost_value:.1f}")
    lines.append("")
    lines.append(f"Average recovery time: {metrics['average_recovery_time']:.2f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    generate_report()
