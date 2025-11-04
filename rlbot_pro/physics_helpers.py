from __future__ import annotations

import math
from collections.abc import Sequence

from rlbot_pro.state import BallState, Vector

GRAVITY = (0.0, 0.0, -650.0)


def integrate_constant_accel(
    pos: Vector, vel: Vector, acc: Vector, dt: float
) -> tuple[Vector, Vector]:
    px, py, pz = pos
    vx, vy, vz = vel
    ax, ay, az = acc
    new_pos = (
        px + vx * dt + 0.5 * ax * dt * dt,
        py + vy * dt + 0.5 * ay * dt * dt,
        pz + vz * dt + 0.5 * az * dt * dt,
    )
    new_vel = (vx + ax * dt, vy + ay * dt, vz + az * dt)
    return new_pos, new_vel


def estimate_ball_bounce(ball: BallState, elasticity: float = 0.6) -> BallState:
    pos = ball.pos
    vel = ball.vel
    if pos[2] <= 0.0 and vel[2] < 0.0:
        vel = (vel[0], vel[1], -vel[2] * elasticity)
        pos = (pos[0], pos[1], 0.0)
    elif pos[2] > 0.0 and vel[2] < 0.0:
        time_to_floor = max(0.0, pos[2] / max(1e-3, -vel[2]))
        new_pos, new_vel = integrate_constant_accel(pos, vel, GRAVITY, time_to_floor)
        new_vel = (new_vel[0], new_vel[1], -new_vel[2] * elasticity)
        pos = (new_pos[0], new_pos[1], max(0.0, new_pos[2]))
        vel = new_vel
    return BallState(pos, vel)


def estimate_backboard_bounce(
    ball: BallState, board_y: float = 5120.0, elasticity: float = 0.6
) -> BallState:
    by = ball.pos[1]
    vy = ball.vel[1]
    if abs(by) < board_y:
        return ball
    if (by > 0 and vy > 0) or (by < 0 and vy < 0):
        new_vy = -vy * elasticity
        new_py = math.copysign(board_y - 10.0, by)
        next_pos = (ball.pos[0], new_py, ball.pos[2])
        next_vel = (ball.vel[0], new_vy, ball.vel[2])
        return BallState(next_pos, next_vel)
    return ball


def time_until_height(
    pos: Vector, vel: Vector, target_z: float, acc: Vector = GRAVITY
) -> float | None:
    _, _, pz = pos
    _, _, vz = vel
    _, _, az = acc
    a = 0.5 * az
    b = vz
    c = pz - target_z
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a) if a != 0 else None
    t2 = (-b + sqrt_disc) / (2 * a) if a != 0 else None
    candidates = [t for t in (t1, t2) if t is not None and t >= 0]
    return min(candidates, default=None)


def cubic_spline(
    points: Sequence[tuple[float, float]] | Sequence[Vector],
    resolution: int = 50,
) -> list[tuple[float, float]]:
    if resolution < 2:
        message = "resolution must be at least 2"
        raise ValueError(message)
    if len(points) < 2:
        message = "at least two points are required"
        raise ValueError(message)
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    n = len(points)
    hs = [xs[i + 1] - xs[i] for i in range(n - 1)]
    if any(h == 0 for h in hs):
        message = "points must have unique x values"
        raise ValueError(message)
    alphas = [0.0] * n
    for i in range(1, n - 1):
        forward_slope = (ys[i + 1] - ys[i]) / hs[i]
        backward_slope = (ys[i] - ys[i - 1]) / hs[i - 1]
        alphas[i] = 3 * (forward_slope - backward_slope)
    ls = [1.0] + [0.0] * (n - 1)
    mus = [0.0] * n
    zs = [0.0] * n
    for i in range(1, n - 1):
        ls[i] = 2 * (xs[i + 1] - xs[i - 1]) - hs[i - 1] * mus[i - 1]
        mus[i] = hs[i] / ls[i]
        zs[i] = (alphas[i] - hs[i - 1] * zs[i - 1]) / ls[i]
    ls[-1] = 1.0
    zs[-1] = 0.0
    cs = [0.0] * n
    bs = [0.0] * (n - 1)
    ds = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        cs[j] = zs[j] - mus[j] * cs[j + 1]
        bs[j] = (ys[j + 1] - ys[j]) / hs[j] - hs[j] * (cs[j + 1] + 2 * cs[j]) / 3
        ds[j] = (cs[j + 1] - cs[j]) / (3 * hs[j])
    samples: list[tuple[float, float]] = []
    step_count = max(1, resolution // (n - 1))
    for i in range(n - 1):
        x0 = xs[i]
        for step in range(step_count):
            t = hs[i] * (step / step_count)
            y = ys[i] + bs[i] * t + cs[i] * t * t + ds[i] * t * t * t
            samples.append((x0 + t, y))
    samples.append((xs[-1], ys[-1]))
    return samples


__all__ = [
    "GRAVITY",
    "integrate_constant_accel",
    "estimate_ball_bounce",
    "estimate_backboard_bounce",
    "time_until_height",
    "cubic_spline",
]
