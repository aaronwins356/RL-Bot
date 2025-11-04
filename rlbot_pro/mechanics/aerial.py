"""Basic aerial controller heuristics."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, magnitude, normalize, subtract
from ..state import GameState


@dataclass(frozen=True)
class AerialParams:
    """Parameters guiding an aerial approach."""

    target: Vector3
    arrival_time: float


def aerial_step(state: GameState, params: AerialParams) -> Controls:
    """Generate controls guiding the car toward an aerial target."""
    to_target = subtract(params.target, state.car.position)
    distance = magnitude(to_target)
    direction = normalize(to_target)
    vertical_error = clamp(params.target.z - state.car.position.z, -1_000.0, 1_000.0)
    time_factor = 0.0
    if params.arrival_time > 0.0:
        distance_over_time = distance / max(params.arrival_time, 1e-3)
        time_factor = clamp(distance_over_time, 0.0, 2_300.0)
    throttle = clamp(time_factor / 2_300.0, 0.0, 1.0)
    pitch = clamp(-direction.z, -1.0, 1.0)
    yaw = clamp(direction.y, -1.0, 1.0)
    roll = clamp(direction.x, -1.0, 1.0)
    return Controls(
        throttle=throttle,
        steer=yaw,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        boost=vertical_error > 0,
        jump=distance > 100.0 and not state.car.on_ground,
        handbrake=False,
    )


__all__ = ["AerialParams", "aerial_step"]
