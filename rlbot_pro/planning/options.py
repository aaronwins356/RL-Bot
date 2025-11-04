"""Planning options for high-level behavior selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..control import Controls
from ..math3d import Vector3
from ..mechanics import (
    AerialMechanic,
    AerialParams,
    AirDribbleMechanic,
    AirDribbleParams,
    CeilingShotMechanic,
    CeilingShotParams,
    DoubleTapMechanic,
    DoubleTapParams,
    FlipResetMechanic,
    FlipResetParams,
    RecoveryMechanic,
    RecoveryParams,
    RecoveryType,
)
from ..sensing import time_to_ball_touch
from ..state import GameState


class OptionType(Enum):
    """Enumerate high-level offensive maneuvers."""

    GROUND_DRIVE = auto()
    AERIAL = auto()
    AIR_DRIBBLE = auto()
    FLIP_RESET = auto()
    CEILING = auto()
    DOUBLE_TAP = auto()
    RECOVERY = auto()


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
    if state.ball.position.y > 4_500.0 and state.ball.velocity.y > 0.0:
        return OptionType.DOUBLE_TAP
    is_ceiling_setup = (
        state.car.on_ground
        and state.ball.position.z < 200.0
        and state.ball.velocity.z > 0.0
    )
    if is_ceiling_setup:
        return OptionType.CEILING
    if not state.car.on_ground and state.car.velocity.z < -300.0:
        return OptionType.RECOVERY
    return OptionType.GROUND_DRIVE


def execute(state: GameState, option_type: OptionType) -> Option:
    """Construct controls for the chosen option."""
    if option_type is OptionType.AERIAL:
        mechanic = AerialMechanic(
            AerialParams(
                intercept=state.ball.position,
                arrival_time=time_to_ball_touch(state),
            )
        )
        mechanic.prep(state)
        return Option(option_type, mechanic.step(state))
    if option_type is OptionType.AIR_DRIBBLE:
        mechanic = AirDribbleMechanic(
            AirDribbleParams(
                carry_offset=Vector3(0.0, 0.0, 120.0),
                target_velocity=1_500.0,
            )
        )
        mechanic.prep(state)
        return Option(option_type, mechanic.step(state))
    if option_type is OptionType.FLIP_RESET:
        mechanic = FlipResetMechanic(
            FlipResetParams(
                target_surface_normal=Vector3(0.0, 0.0, 1.0),
                commit_time=time_to_ball_touch(state),
            )
        )
        mechanic.prep(state)
        return Option(option_type, mechanic.step(state))
    if option_type is OptionType.CEILING:
        mechanic = CeilingShotMechanic(
            CeilingShotParams(
                carry_target=Vector3(state.ball.position.x, state.ball.position.y, 2_000.0),
                detach_height=1_800.0,
                detach_time=time_to_ball_touch(state),
                flip_window=(0.15, 0.35),
            )
        )
        mechanic.prep(state)
        return Option(option_type, mechanic.step(state))
    if option_type is OptionType.DOUBLE_TAP:
        mechanic = DoubleTapMechanic(
            DoubleTapParams(
                backboard_y=5_120.0,
                restitution=0.65,
                first_touch_speed=1_600.0,
                second_arrival_time=0.9,
            )
        )
        mechanic.prep(state)
        return Option(option_type, mechanic.step(state))
    if option_type is OptionType.RECOVERY:
        mechanic = RecoveryMechanic(RecoveryParams(strategy=RecoveryType.UPRIGHT))
        mechanic.prep(state)
        return Option(option_type, mechanic.step(state))
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
