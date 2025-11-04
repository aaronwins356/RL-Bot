from __future__ import annotations

from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.math3d import clamp, normalize, vec_sub
from rlbot_pro.sensing import is_near_wall
from rlbot_pro.state import GameState, Vector


@dataclass(frozen=True, slots=True)
class Params:
    detach_height: float = 1800.0
    wall_margin: float = 300.0
    dodge_time: float = 1.0


DEFAULT_PARAMS = Params()


def prep(gs: GameState, params: Params = DEFAULT_PARAMS) -> None:
    if not is_near_wall(gs.car.pos, params.wall_margin):
        return


def step(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> Controls:
    target_pos = target or (
        gs.ball.pos[0],
        gs.ball.pos[1],
        params.detach_height,
    )
    to_target = normalize(vec_sub(target_pos, gs.car.pos))
    throttle = clamp(0.8 + to_target[1] * 0.2, -1.0, 1.0)
    pitch = clamp(-to_target[2], -1.0, 1.0)
    yaw = clamp(to_target[0], -1.0, 1.0)
    boost = bool(gs.car.boost > 0 and gs.car.pos[2] < params.detach_height - 100)
    jump = bool(gs.car.pos[2] > params.detach_height - 50 and gs.car.has_flip)
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
    height_reached = gs.car.pos[2] >= params.detach_height
    stable = abs(gs.car.vel[2]) < 50
    return height_reached and stable


def is_invalid(
    gs: GameState, params: Params = DEFAULT_PARAMS, *, target: Vector | None = None
) -> bool:
    if gs.ball.pos[2] < 600:
        return True
    if not is_near_wall(gs.car.pos, params.wall_margin):
        return True
    return False


__all__ = ["Params", "DEFAULT_PARAMS", "prep", "step", "is_complete", "is_invalid"]
