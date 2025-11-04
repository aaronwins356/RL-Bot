from __future__ import annotations

from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.math3d import clamp, magnitude, normalize, vec_add, vec_scale, vec_sub
from rlbot_pro.sensing import is_car_grounded
from rlbot_pro.state import GameState, Vector


@dataclass(frozen=True, slots=True)
class Params:
    carry_height: float = 750.0
    boost_usage: float = 0.7
    approach_distance: float = 900.0


DEFAULT_PARAMS = Params()


def prep(gs: GameState, params: Params = DEFAULT_PARAMS) -> None:
    return None


def _target_point(gs: GameState, params: Params) -> Vector:
    car_forward = normalize(gs.car.forward)
    offset = vec_scale(car_forward, params.approach_distance * 0.2)
    return vec_add(gs.ball.pos, offset)


def step(gs: GameState, params: Params = DEFAULT_PARAMS) -> Controls:
    target = _target_point(gs, params)
    to_target = normalize(vec_sub(target, gs.car.pos))
    height_error = params.carry_height - gs.ball.pos[2]
    throttle = clamp(0.5 + to_target[1], -1.0, 1.0)
    pitch = clamp(-height_error / params.carry_height, -1.0, 1.0)
    yaw = clamp(to_target[0], -1.0, 1.0)
    boost_ready = gs.car.boost > 10
    height_ok = height_error > -100
    vertical_limit = gs.car.vel[2] < 400 * params.boost_usage
    boost = bool(boost_ready and height_ok and vertical_limit)
    jump = not is_car_grounded(gs) and height_error > 50 and gs.car.has_flip
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
    return gs.ball.pos[2] > params.carry_height and gs.car.pos[2] > gs.ball.pos[2] - 50


def is_invalid(gs: GameState, params: Params = DEFAULT_PARAMS) -> bool:
    if gs.car.boost < 20:
        return True
    if magnitude(vec_sub(gs.ball.pos, gs.car.pos)) > params.approach_distance:
        return True
    return False


__all__ = ["Params", "DEFAULT_PARAMS", "prep", "step", "is_complete", "is_invalid"]
