from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

Vector = tuple[float, float, float]


def _as_vector(values: Iterable[float]) -> Vector:
    data = tuple(float(v) for v in values)
    if len(data) != 3:
        message = "Vectors must contain exactly three components"
        raise ValueError(message)
    return data


@dataclass(frozen=True, slots=True)
class BallState:
    pos: Vector
    vel: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _as_vector(self.pos))
        object.__setattr__(self, "vel", _as_vector(self.vel))

    def predict(self, dt: float) -> BallState:
        px, py, pz = self.pos
        vx, vy, vz = self.vel
        return BallState((px + vx * dt, py + vy * dt, pz + vz * dt), self.vel)


@dataclass(frozen=True, slots=True)
class CarState:
    pos: Vector
    vel: Vector
    ang_vel: Vector
    forward: Vector
    up: Vector
    boost: float
    has_flip: bool
    on_ground: bool
    time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _as_vector(self.pos))
        object.__setattr__(self, "vel", _as_vector(self.vel))
        object.__setattr__(self, "ang_vel", _as_vector(self.ang_vel))
        object.__setattr__(self, "forward", _as_vector(self.forward))
        object.__setattr__(self, "up", _as_vector(self.up))
        boost = max(0.0, min(100.0, float(self.boost)))
        object.__setattr__(self, "boost", boost)

    @property
    def location(self) -> Vector:
        return self.pos

    @property
    def velocity(self) -> Vector:
        return self.vel

    def with_time(self, time: float) -> CarState:
        return CarState(
            pos=self.pos,
            vel=self.vel,
            ang_vel=self.ang_vel,
            forward=self.forward,
            up=self.up,
            boost=self.boost,
            has_flip=self.has_flip,
            on_ground=self.on_ground,
            time=time,
        )


@dataclass(frozen=True, slots=True)
class GameState:
    ball: BallState
    car: CarState
    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            message = "dt must be positive"
            raise ValueError(message)

    def advance(self, pos: Vector, vel: Vector, dt: float | None = None) -> GameState:
        new_dt = float(self.dt if dt is None else dt)
        if new_dt <= 0.0:
            message = "dt must be positive"
            raise ValueError(message)
        return GameState(ball=BallState(pos, vel), car=self.car, dt=new_dt)


__all__ = ["Vector", "BallState", "CarState", "GameState"]
