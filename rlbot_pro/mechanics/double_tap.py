from __future__ import annotations

from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.math3d import clamp, magnitude, normalize, vec_sub
from rlbot_pro.physics_helpers import estimate_backboard_bounce
from rlbot_pro.state import GameState, Vector


@dataclass(frozen=True, slots=True)
class Params:
    bounce_height: float = 800.0
    min_backboard_y: float = 4800.0
    finish_distance: float = 150.0


DEFAULT_PARAMS = Params()


def prep(gs: GameState, params: Params = DEFAULT_PARAMS) -> None:
    return None


def _predict_target(gs: GameState, params: Params) -> Vector:
    predicted = estimate_backboard_bounce(gs.ball)
    return (
        predicted.pos[0],
        predicted.pos[1],
        max(params.bounce_height, predicted.pos[2]),
    )


def step(gs: GameState, params: Params = DEFAULT_PARAMS) -> Controls:
    target = _predict_target(gs, params)
    to_target = normalize(vec_sub(target, gs.car.pos))
    distance = magnitude(vec_sub(target, gs.car.pos))
    throttle = clamp(0.4 + to_target[1] * 0.6, -1.0, 1.0)
    vertical_delta = target[2] - gs.car.pos[2]
    pitch = clamp(vertical_delta / params.bounce_height, -1.0, 1.0)
    yaw = clamp(to_target[0], -1.0, 1.0)
    boost = gs.car.boost > 20 and distance > params.finish_distance
    jump = distance < params.finish_distance and gs.car.has_flip
    return Controls(
        throttle=throttle,
        steer=yaw,
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
        boost=boost,
        jump=jump,
        handbrake=False,
    )


def is_complete(gs: GameState, params: Params = DEFAULT_PARAMS) -> bool:
    target = _predict_target(gs, params)
    return magnitude(vec_sub(target, gs.car.pos)) < params.finish_distance


def is_invalid(gs: GameState, params: Params = DEFAULT_PARAMS) -> bool:
    if abs(gs.ball.pos[1]) < params.min_backboard_y:
        return True
    return False


__all__ = ["Params", "DEFAULT_PARAMS", "prep", "step", "is_complete", "is_invalid"]
