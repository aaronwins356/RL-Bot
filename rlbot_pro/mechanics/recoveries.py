from __future__ import annotations

from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.math3d import clamp
from rlbot_pro.sensing import is_car_grounded
from rlbot_pro.state import GameState


@dataclass(frozen=True, slots=True)
class Params:
    roll_gain: float = 0.6
    pitch_gain: float = 0.7
    yaw_gain: float = 0.5


DEFAULT_PARAMS = Params()


def prep(gs: GameState, params: Params = DEFAULT_PARAMS) -> None:
    return None


def step(gs: GameState, params: Params = DEFAULT_PARAMS) -> Controls:
    roll = clamp(-gs.car.up[0] * params.roll_gain, -1.0, 1.0)
    pitch = clamp(gs.car.up[2] - 1.0, -1.0, 1.0) * params.pitch_gain
    yaw = clamp(-gs.car.up[1] * params.yaw_gain, -1.0, 1.0)
    throttle = 0.5 if not is_car_grounded(gs) else 0.0
    return Controls(
        throttle=throttle,
        steer=yaw,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        boost=False,
        jump=False,
        handbrake=False,
    )


def is_complete(gs: GameState, params: Params = DEFAULT_PARAMS) -> bool:
    return is_car_grounded(gs)


def is_invalid(gs: GameState, params: Params = DEFAULT_PARAMS) -> bool:
    return False


__all__ = ["Params", "DEFAULT_PARAMS", "prep", "step", "is_complete", "is_invalid"]
