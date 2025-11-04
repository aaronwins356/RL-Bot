from __future__ import annotations

from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.math3d import clamp, magnitude, normalize, vec_sub
from rlbot_pro.state import GameState, Vector


@dataclass(frozen=True, slots=True)
class Params:
    target_height: float = 800.0
    contact_distance: float = 90.0
    boost_commit: float = 40.0


DEFAULT_PARAMS = Params()


def prep(gs: GameState, params: Params = DEFAULT_PARAMS) -> None:
    return None


def _alignment(gs: GameState, target: Vector) -> float:
    car_forward = normalize(gs.car.forward)
    to_target = normalize(vec_sub(target, gs.car.pos))
    return (
        car_forward[0] * to_target[0]
        + car_forward[1] * to_target[1]
        + car_forward[2] * to_target[2]
    )


def step(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> Controls:
    target_point = target or gs.ball.pos
    alignment = _alignment(gs, target_point)
    distance = magnitude(vec_sub(target_point, gs.car.pos))
    boost = gs.car.boost > params.boost_commit and alignment > 0.6
    throttle = clamp(0.3 + alignment * 0.7, -1.0, 1.0)
    vertical_delta = target_point[2] - gs.car.pos[2]
    pitch = clamp(vertical_delta / params.target_height, -1.0, 1.0)
    horizontal_delta = target_point[0] - gs.car.pos[0]
    yaw = clamp(horizontal_delta / max(1.0, distance), -1.0, 1.0)
    jump = distance < params.contact_distance and gs.car.has_flip
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


def is_complete(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> bool:
    target_point = target or gs.ball.pos
    return magnitude(vec_sub(target_point, gs.car.pos)) < params.contact_distance / 2


def is_invalid(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> bool:
    if gs.ball.pos[2] < params.target_height * 0.5:
        return True
    if gs.car.boost < 10:
        return True
    return False


__all__ = ["Params", "DEFAULT_PARAMS", "prep", "step", "is_complete", "is_invalid"]
