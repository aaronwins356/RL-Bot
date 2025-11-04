from __future__ import annotations

from pathlib import Path

from telemetry.metrics import TelemetryMetrics
from telemetry.telemetry import TelemetryRecord, TelemetryWriter


def test_telemetry_writer_and_metrics(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    writer = TelemetryWriter(path, rotate=False)
    record = TelemetryRecord(
        timestamp=0.0,
        car_pos=(0.0, 0.0, 0.0),
        car_vel=(0.0, 0.0, 0.0),
        ball_pos=(0.0, 0.0, 100.0),
        ball_vel=(0.0, 0.0, 0.0),
        boost=100.0,
        option_kind="AERIAL",
        option_esv=1.0,
        option_desc="test",
        completed=True,
        failed=False,
    )
    writer.write(record)
    writer.close()
    content = path.read_text().strip().splitlines()
    assert content[0].startswith("timestamp")
    assert "AERIAL" in content[1]
    metrics = TelemetryMetrics.from_csv(path)
    data = metrics.to_dict()
    assert data["rows"] == 1
    assert data["conversion_rate"]["AERIAL"] == 1.0
