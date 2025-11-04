"""Ceiling movement helper."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp
from ..state import GameState


@dataclass(frozen=True)
class CeilingShotParams:
    """Configuration for a ceiling setup."""

    drop_point: Vector3
    release_time: float


def ceiling_shot_step(state: GameState, params: CeilingShotParams) -> Controls:
    """Generate a ceiling setup control scheme."""
    vertical_delta = params.drop_point.z - state.car.position.z
    vertical_bias = clamp(vertical_delta / 1_000.0, -1.0, 1.0)
    lateral_delta = params.drop_point.y - state.car.position.y
    lateral_bias = clamp(lateral_delta / 1_000.0, -1.0, 1.0)
    should_release = params.release_time <= 0.0 and state.car.on_ground
    return Controls(
        throttle=0.5,
        steer=lateral_bias,
        pitch=-vertical_bias,
        yaw=lateral_bias,
        roll=0.0,
        boost=False,
        jump=should_release,
        handbrake=False,
    )


__all__ = ["CeilingShotParams", "ceiling_shot_step"]
