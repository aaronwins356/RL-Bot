"""Planning options for high-level behavior selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..control import Controls
from ..math3d import Vector3
from ..mechanics.aerial import AerialParams, aerial_step
from ..mechanics.air_dribble import AirDribbleParams, air_dribble_step
from ..mechanics.ceiling import CeilingShotParams, ceiling_shot_step
from ..mechanics.flip_reset import FlipResetParams, flip_reset_step
from ..sensing import time_to_ball_touch
from ..state import GameState


class OptionType(Enum):
    """Enumerate high-level offensive maneuvers."""

    GROUND_DRIVE = auto()
    AERIAL = auto()
    AIR_DRIBBLE = auto()
    FLIP_RESET = auto()
    CEILING = auto()


@dataclass(frozen=True)
class Option:
    """Plan output container bundling the decision and controls."""

    option_type: OptionType
    controls: Controls


def evaluate(state: GameState) -> OptionType:
    """Select an option based on game context."""
    if state.ball.position.z > 1_500.0:
        return OptionType.AERIAL
    if not state.car.on_ground and state.ball.position.z > 500.0:
        return OptionType.AIR_DRIBBLE
    if not state.car.on_ground and state.car.has_jump:
        return OptionType.FLIP_RESET
    is_ceiling_setup = (
        state.car.on_ground
        and state.ball.position.z < 200.0
        and state.ball.velocity.z > 0.0
    )
    if is_ceiling_setup:
        return OptionType.CEILING
    return OptionType.GROUND_DRIVE


def execute(state: GameState, option_type: OptionType) -> Option:
    """Construct controls for the chosen option."""
    if option_type is OptionType.AERIAL:
        aerial_params = AerialParams(
            target=state.ball.position,
            arrival_time=time_to_ball_touch(state),
        )
        controls = aerial_step(state, aerial_params)
        return Option(option_type, controls)
    if option_type is OptionType.AIR_DRIBBLE:
        dribble_params = AirDribbleParams(
            carry_offset=Vector3(0.0, 0.0, 120.0),
            target_velocity=1_500.0,
        )
        controls = air_dribble_step(state, dribble_params)
        return Option(option_type, controls)
    if option_type is OptionType.FLIP_RESET:
        flip_reset_params = FlipResetParams(
            target_surface_normal=Vector3(0.0, 0.0, 1.0),
            commit_time=time_to_ball_touch(state),
        )
        controls = flip_reset_step(state, flip_reset_params)
        return Option(option_type, controls)
    if option_type is OptionType.CEILING:
        ceiling_params = CeilingShotParams(
            drop_point=Vector3(state.ball.position.x, state.ball.position.y, 2_000.0),
            release_time=time_to_ball_touch(state),
        )
        controls = ceiling_shot_step(state, ceiling_params)
        return Option(option_type, controls)
    controls = Controls(
        throttle=0.7,
        steer=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        boost=False,
        jump=False,
        handbrake=False,
    )
    return Option(option_type, controls)


__all__ = ["OptionType", "Option", "evaluate", "execute"]
