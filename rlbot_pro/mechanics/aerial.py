from __future__ import annotations

import math
from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.math3d import clamp, magnitude, normalize, vec_sub
from rlbot_pro.sensing import forward_alignment, time_to_ball_touch
from rlbot_pro.state import GameState, Vector


@dataclass(frozen=True, slots=True)
class Params:
    boost_threshold: float = 20.0
    alignment_limit: float = math.radians(35)
    finish_distance: float = 120.0


DEFAULT_PARAMS = Params()


def prep(gs: GameState, params: Params = DEFAULT_PARAMS) -> None:  # noqa: D401
    """No-op prep to satisfy mechanic contract."""


def _compute_orientation(gs: GameState, target: Vector) -> tuple[float, float, float]:
    car_forward = normalize(gs.car.forward)
    to_target = normalize(vec_sub(target, gs.car.pos))
    yaw_error = clamp(to_target[0] - car_forward[0], -1.0, 1.0)
    pitch_error = clamp(car_forward[2] - to_target[2], -1.0, 1.0)
    roll_correction = clamp(-gs.car.up[0], -1.0, 1.0)
    return yaw_error, pitch_error, roll_correction


def step(
    gs: GameState,
    params: Params = DEFAULT_PARAMS,
    *,
    target: Vector | None = None,
) -> Controls:
    dest = target or gs.ball.pos
    yaw_error, pitch_error, roll_correction = _compute_orientation(gs, dest)
    distance = magnitude(vec_sub(dest, gs.car.pos))
    alignment = forward_alignment(gs)
    boost_ready = gs.car.boost > params.boost_threshold
    boost = bool(boost_ready and alignment > 0.5 and distance > params.finish_distance)
    throttle = 1.0 if boost else clamp(0.3 + alignment * 0.6, 0.0, 1.0)
    jump = time_to_ball_touch(gs) < 0.5 and not gs.car.on_ground
    return Controls(
        throttle=throttle,
        steer=yaw_error,
        pitch=-pitch_error,
        yaw=yaw_error,
        roll=roll_correction,
        boost=boost,
        jump=jump,
        handbrake=False,
    )


def is_complete(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> bool:
    dest = target or gs.ball.pos
    distance = magnitude(vec_sub(dest, gs.car.pos))
    return distance < params.finish_distance


def is_invalid(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> bool:
    if gs.car.boost < 5.0:
        return True
    if gs.ball.pos[2] < 200.0:
        return True
    alignment = forward_alignment(gs)
    return alignment < math.cos(params.alignment_limit)


__all__ = ["Params", "DEFAULT_PARAMS", "prep", "step", "is_complete", "is_invalid"]
