"""Core game state data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

from .math3d import Vector3


@dataclass(frozen=True)
class BallState:
    """Ball kinematic state."""

    position: Vector3
    velocity: Vector3


@dataclass(frozen=True)
class CarState:
    """Car kinematic state and status flags."""

    position: Vector3
    velocity: Vector3
    boost: float
    has_jump: bool
    is_demolished: bool
    on_ground: bool
    forward: Vector3 = field(default_factory=lambda: Vector3(1.0, 0.0, 0.0))
    up: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 1.0))
    angular_velocity: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))


@dataclass(frozen=True)
class GameState:
    """Aggregated game state for a single agent."""

    ball: BallState
    car: CarState
    time_remaining: float | None
