"""Primitive flip-reset execution."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, dot, normalize, subtract
from ..state import GameState


@dataclass(frozen=True)
class FlipResetParams:
    """Parameters directing a flip reset attempt."""

    target_surface_normal: Vector3
    commit_time: float


def flip_reset_step(state: GameState, params: FlipResetParams) -> Controls:
    """Attempt to align the car with the ball for a flip reset touch."""
    relative = subtract(state.ball.position, state.car.position)
    normal = normalize(params.target_surface_normal)
    alignment = dot(normalize(relative), normal)
    should_flip = alignment > 0.6 and state.car.has_jump and params.commit_time <= 0.0
    pitch = clamp(-relative.z / 500.0, -1.0, 1.0)
    yaw = clamp(relative.y / 500.0, -1.0, 1.0)
    roll = clamp(relative.x / 500.0, -1.0, 1.0)
    return Controls(
        throttle=1.0,
        steer=yaw,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        boost=False,
        jump=should_flip,
        handbrake=False,
    )


__all__ = ["FlipResetParams", "flip_reset_step"]
