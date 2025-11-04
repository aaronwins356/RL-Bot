"""Simple air-dribble maintenance controls."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, subtract
from ..state import GameState


@dataclass(frozen=True)
class AirDribbleParams:
    """Parameters guiding an air dribble."""

    carry_offset: Vector3
    target_velocity: float


def air_dribble_step(state: GameState, params: AirDribbleParams) -> Controls:
    """Bias the car to remain under the ball while modulating speed."""
    desired_position = subtract(state.ball.position, params.carry_offset)
    lateral_error = subtract(desired_position, state.car.position)
    throttle = clamp(params.target_velocity / 2_300.0, 0.0, 1.0)
    steer = clamp(lateral_error.y / 1_000.0, -1.0, 1.0)
    pitch = clamp(-lateral_error.z / 1_000.0, -1.0, 1.0)
    return Controls(
        throttle=throttle,
        steer=steer,
        pitch=pitch,
        yaw=steer,
        roll=0.0,
        boost=params.target_velocity > 1_400.0,
        jump=not state.car.on_ground,
        handbrake=False,
    )


__all__ = ["AirDribbleParams", "air_dribble_step"]
