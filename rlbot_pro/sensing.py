from __future__ import annotations

import math

from rlbot_pro.math3d import distance, magnitude, normalize, signed_angle_2d, vec_sub
from rlbot_pro.state import GameState, Vector

NET_Y = 5120.0
FIELD_X = 4096.0
FIELD_Y = 5120.0
FIELD_Z = 2044.0


def angle_to_target(origin: Vector, target: Vector) -> float:
    direction = vec_sub(target, origin)
    facing = (0.0, 1.0, 0.0)
    return signed_angle_2d(facing, direction)


def angle_between_vectors(a: Vector, b: Vector) -> float:
    return signed_angle_2d(a, b)


def time_to_ball_touch(gs: GameState) -> float:
    relative_pos = distance(gs.car.pos, gs.ball.pos)
    relative_vel = magnitude(vec_sub(gs.ball.vel, gs.car.vel))
    if relative_vel < 1e-3:
        return float("inf")
    return max(0.0, relative_pos / relative_vel)


def time_to_height(gs: GameState, target_z: float) -> float:
    vz = gs.car.vel[2]
    dz = target_z - gs.car.pos[2]
    if abs(vz) < 1e-3:
        return float("inf") if dz > 0 else 0.0
    t = dz / vz
    return t if t >= 0 else float("inf")


def is_car_grounded(gs: GameState) -> bool:
    return gs.car.on_ground or gs.car.pos[2] <= 5.0


def is_near_wall(pos: Vector, margin: float = 200.0) -> bool:
    return abs(pos[0]) > FIELD_X - margin or abs(pos[1]) > FIELD_Y - margin


def has_shot_line(gs: GameState, net_y: float = NET_Y) -> bool:
    ball_to_net = (gs.ball.pos[0], net_y - gs.ball.pos[1], gs.ball.pos[2])
    car_to_ball = vec_sub(gs.ball.pos, gs.car.pos)
    angle = abs(signed_angle_2d(car_to_ball, ball_to_net))
    return angle < math.radians(30)


def forward_alignment(gs: GameState) -> float:
    car_forward = normalize(gs.car.forward)
    to_ball = normalize(vec_sub(gs.ball.pos, gs.car.pos))
    return 1.0 - min(1.0, abs(signed_angle_2d(car_forward, to_ball)) / math.pi)


def pressure_factor(ball_pos: Vector) -> float:
    return min(1.0, max(0.0, (NET_Y - abs(ball_pos[1])) / NET_Y))


__all__ = [
    "angle_to_target",
    "time_to_ball_touch",
    "time_to_height",
    "is_car_grounded",
    "is_near_wall",
    "has_shot_line",
    "forward_alignment",
    "pressure_factor",
]
