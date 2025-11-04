from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TypedDict


class TelemetrySummary(TypedDict):
    conversion_rate: dict[str, float]
    average_recovery_time: float
    average_boost_per_attempt: dict[str, float]
    rows: int


class TelemetryMetrics:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    @classmethod
    def from_csv(cls, path: str | Path) -> TelemetryMetrics:
        with Path(path).open(newline="") as fh:
            reader = csv.DictReader(fh)
            return cls(list(reader))

    def conversion_rate_per_mechanic(self) -> dict[str, float]:
        success: dict[str, int] = defaultdict(int)
        attempts: dict[str, int] = defaultdict(int)
        for row in self.rows:
            kind = row["option"]
            attempts[kind] += 1
            if row.get("completed") == "1":
                success[kind] += 1
        return {
            kind: (success[kind] / attempts[kind]) if attempts[kind] else 0.0 for kind in attempts
        }

    def average_recovery_time(self) -> float:
        recovery_starts: list[float] = []
        recovery_ends: list[float] = []
        for row in self.rows:
            if row["option"].upper() == "RECOVERY":
                recovery_starts.append(float(row["timestamp"]))
            elif recovery_starts and float(row["timestamp"]) > recovery_starts[-1]:
                recovery_ends.append(float(row["timestamp"]))
        if not recovery_starts or not recovery_ends:
            return 0.0
        durations = [
            end - start for start, end in zip(recovery_starts, recovery_ends, strict=False)
        ]
        return mean(durations) if durations else 0.0

    def average_boost_per_attempt(self) -> dict[str, float]:
        totals: dict[str, list[float]] = defaultdict(list)
        for row in self.rows:
            totals[row["option"]].append(float(row["boost"]))
        return {kind: mean(values) if values else 0.0 for kind, values in totals.items()}

    def to_dict(self) -> TelemetrySummary:
        return {
            "conversion_rate": self.conversion_rate_per_mechanic(),
            "average_recovery_time": self.average_recovery_time(),
            "average_boost_per_attempt": self.average_boost_per_attempt(),
            "rows": len(self.rows),
        }


__all__ = ["TelemetryMetrics", "TelemetrySummary"]
