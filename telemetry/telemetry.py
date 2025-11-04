from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

from rlbot_pro.control import Controls
from rlbot_pro.planning.options import Option
from rlbot_pro.state import GameState


@dataclass(slots=True)
class TelemetryRecord:
    timestamp: float
    car_pos: tuple[float, float, float]
    car_vel: tuple[float, float, float]
    ball_pos: tuple[float, float, float]
    ball_vel: tuple[float, float, float]
    boost: float
    option_kind: str
    option_esv: float
    option_desc: str
    completed: bool
    failed: bool

    @classmethod
    def from_state(cls, gs: GameState, option: Option, controls: Controls) -> TelemetryRecord:
        _ = controls
        return cls(
            timestamp=gs.car.time,
            car_pos=gs.car.pos,
            car_vel=gs.car.vel,
            ball_pos=gs.ball.pos,
            ball_vel=gs.ball.vel,
            boost=gs.car.boost,
            option_kind=option.kind.name,
            option_esv=option.esv,
            option_desc=option.description,
            completed=False,
            failed=False,
        )

    def to_row(self) -> list[str]:
        return [
            f"{self.timestamp:.3f}",
            *[f"{v:.3f}" for v in self.car_pos],
            *[f"{v:.3f}" for v in self.car_vel],
            *[f"{v:.3f}" for v in self.ball_pos],
            *[f"{v:.3f}" for v in self.ball_vel],
            f"{self.boost:.1f}",
            self.option_kind,
            f"{self.option_esv:.3f}",
            self.option_desc,
            "1" if self.completed else "0",
            "1" if self.failed else "0",
        ]


class TelemetryWriter:
    header = [
        "timestamp",
        "car_x",
        "car_y",
        "car_z",
        "car_vx",
        "car_vy",
        "car_vz",
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_vx",
        "ball_vy",
        "ball_vz",
        "boost",
        "option",
        "option_esv",
        "option_desc",
        "completed",
        "failed",
    ]

    def __init__(self, path: str | Path, rotate: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if rotate and self.path.exists():
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            self.path.replace(backup)
        self._file: TextIO = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)
        self._file.flush()

    def write(self, record: TelemetryRecord) -> None:
        self._writer.writerow(record.to_row())
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["TelemetryRecord", "TelemetryWriter"]
