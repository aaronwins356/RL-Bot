from dataclasses import dataclass
from typing import Tuple

Vector = Tuple[float, float, float]


@dataclass(frozen=True)
class BallState:
    """Represents the state of the ball."""

    pos: Vector
    vel: Vector
    ang_vel: Vector = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CarState:
    """Represents the state of a car."""

    pos: Vector
    vel: Vector
    ang_vel: Vector
    forward: Vector
    up: Vector
    boost: float
    has_flip: bool
    on_ground: bool
    time: float


@dataclass(frozen=True)
class GameState:
    """Represents the overall game state."""

    ball: BallState
    car: CarState
    dt: float
