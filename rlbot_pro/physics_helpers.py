"""Deterministic physics helper utilities for mechanics planning."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

from .control import Controls
from .math3d import Vector3, add, clamp, dot, magnitude, normalize, scale, subtract
from .state import BallState, CarState


GRAVITY = Vector3(0.0, 0.0, -650.0)
CAR_MAX_SPEED = 2300.0
CAR_MAX_ACCELERATION = 1000.0
BOOST_ACCELERATION = 991.666
BOOST_CONSUMPTION_PER_SECOND = 33.3


def advance_linear(position: Vector3, velocity: Vector3, acceleration: Vector3, dt: float) -> tuple[Vector3, Vector3]:
    """Advance a kinematic state forward by ``dt`` seconds."""

    new_velocity = add(velocity, scale(acceleration, dt))
    midpoint_velocity = add(velocity, new_velocity)
    midpoint_velocity = scale(midpoint_velocity, 0.5)
    new_position = add(position, scale(midpoint_velocity, dt))
    return new_position, new_velocity


def apply_ground_bounce(position: Vector3, velocity: Vector3, restitution: float = 0.6) -> tuple[Vector3, Vector3]:
    """Reflect the vertical component if the object would hit the ground."""

    if position.z >= 0.0:
        return position, velocity
    clamped_position = Vector3(position.x, position.y, 0.0)
    reflected_velocity = Vector3(velocity.x, velocity.y, -velocity.z * restitution)
    return clamped_position, reflected_velocity


def advance_ball(ball: BallState, dt: float) -> BallState:
    """Advance the ball state using simple ballistic physics."""

    position, velocity = advance_linear(ball.position, ball.velocity, GRAVITY, dt)
    position, velocity = apply_ground_bounce(position, velocity)
    return BallState(position=position, velocity=velocity)


def advance_car(car: CarState, controls: Controls, dt: float) -> CarState:
    """Integrate a simplified car model focusing on speed and vertical motion."""

    throttle_accel = controls.throttle * CAR_MAX_ACCELERATION
    forward = normalize(car.forward)
    lateral_speed = dot(car.velocity, forward)
    target_speed = clamp(lateral_speed + throttle_accel * dt, -CAR_MAX_SPEED, CAR_MAX_SPEED)
    forward_velocity = scale(forward, target_speed)

    vertical_velocity = car.velocity.z + controls.pitch * CAR_MAX_ACCELERATION * dt
    vertical_velocity += GRAVITY.z * dt

    new_velocity = Vector3(forward_velocity.x, forward_velocity.y, vertical_velocity)
    new_position = add(car.position, scale(new_velocity, dt))

    boost = car.boost
    if controls.boost and boost > 0.0:
        boost_spent = min(boost, BOOST_CONSUMPTION_PER_SECOND * dt)
        boost -= boost_spent
        boost_velocity = scale(forward, BOOST_ACCELERATION * dt)
        new_velocity = add(new_velocity, boost_velocity)
        new_position = add(car.position, scale(new_velocity, dt))

    on_ground = new_position.z <= 17.0
    if on_ground:
        new_position = Vector3(new_position.x, new_position.y, 17.0)
        new_velocity = Vector3(new_velocity.x * 0.9, new_velocity.y * 0.9, 0.0)

    jump_available = car.has_jump
    if controls.jump and jump_available:
        jump_available = False
        new_velocity = add(new_velocity, Vector3(0.0, 0.0, 300.0))
        on_ground = False

    new_forward = normalize(new_velocity)
    if magnitude(new_forward) <= 1e-5:
        new_forward = forward

    return CarState(
        position=new_position,
        velocity=new_velocity,
        boost=boost,
        has_jump=jump_available,
        is_demolished=False,
        on_ground=on_ground,
        forward=new_forward,
        up=car.up,
        angular_velocity=car.angular_velocity,
    )


def estimate_time_to_cover(distance_m: float, speed: float, accel: float) -> float:
    """Solve for the travel time given distance, current speed, and constant acceleration."""

    if distance_m <= 0.0:
        return 0.0
    if accel <= 1e-6:
        if speed <= 1e-6:
            return float("inf")
        return distance_m / speed
    discriminant = speed * speed + 2.0 * accel * distance_m
    return (sqrt(discriminant) - speed) / accel


def estimate_required_speed(distance_m: float, time_s: float, max_speed: float = CAR_MAX_SPEED) -> float:
    """Return the speed needed to cover ``distance_m`` in ``time_s`` seconds."""

    if time_s <= 1e-6:
        return max_speed
    return clamp(distance_m / time_s, 0.0, max_speed)


def predict_backboard_bounce(
    position: Vector3,
    velocity: Vector3,
    backboard_y: float,
    restitution: float = 0.6,
) -> tuple[Vector3, Vector3, float]:
    """Predict a planar bounce on the backboard assuming constant velocity until impact."""

    travel_distance = backboard_y - position.y
    if abs(velocity.y) <= 1e-3:
        return position, Vector3(0.0, 0.0, 0.0), float("inf")
    time_to_board = travel_distance / velocity.y
    impact_position = add(position, scale(velocity, time_to_board))
    reflected_velocity = Vector3(velocity.x, -velocity.y * restitution, velocity.z)
    return impact_position, reflected_velocity, time_to_board


@dataclass(frozen=True)
class CubicSpline:
    """Natural cubic spline interpolant for Vector3 control points."""

    times: Sequence[float]
    points: Sequence[Vector3]
    _second_derivatives_x: tuple[float, ...]
    _second_derivatives_y: tuple[float, ...]
    _second_derivatives_z: tuple[float, ...]

    def __init__(self, times: Sequence[float], points: Sequence[Vector3]) -> None:
        if len(times) != len(points):
            raise ValueError("Times and points must have matching length")
        if len(times) < 2:
            raise ValueError("At least two knots are required for interpolation")
        if any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
            raise ValueError("Times must be strictly increasing")
        object.__setattr__(self, "times", tuple(times))
        object.__setattr__(self, "points", tuple(points))
        second_x = self._solve_second_derivatives([p.x for p in points])
        second_y = self._solve_second_derivatives([p.y for p in points])
        second_z = self._solve_second_derivatives([p.z for p in points])
        object.__setattr__(self, "_second_derivatives_x", tuple(second_x))
        object.__setattr__(self, "_second_derivatives_y", tuple(second_y))
        object.__setattr__(self, "_second_derivatives_z", tuple(second_z))

    def _solve_second_derivatives(self, values: Sequence[float]) -> list[float]:
        n = len(self.times)
        second = [0.0] * n
        u = [0.0] * (n - 1)
        for i in range(1, n - 1):
            sig = (self.times[i] - self.times[i - 1]) / (self.times[i + 1] - self.times[i - 1])
            p = sig * second[i - 1] + 2.0
            second[i] = (sig - 1.0) / p
            delta = (
                (values[i + 1] - values[i]) / (self.times[i + 1] - self.times[i])
                - (values[i] - values[i - 1]) / (self.times[i] - self.times[i - 1])
            )
            u[i] = (6.0 * delta / (self.times[i + 1] - self.times[i - 1]) - sig * u[i - 1]) / p
        for k in range(n - 2, -1, -1):
            second[k] = second[k] * second[k + 1] + u[k]
        return second

    def evaluate(self, t: float) -> Vector3:
        """Evaluate the spline at the specified time."""

        times = self.times
        if t <= times[0]:
            return self.points[0]
        if t >= times[-1]:
            return self.points[-1]
        idx = bisect_right(times, t) - 1
        idx = max(0, min(idx, len(times) - 2))
        h = times[idx + 1] - times[idx]
        a = (times[idx + 1] - t) / h
        b = (t - times[idx]) / h
        sx = (
            a * self.points[idx].x
            + b * self.points[idx + 1].x
            + ((a * a * a - a) * self._second_derivatives_x[idx]
            + (b * b * b - b) * self._second_derivatives_x[idx + 1])
            * (h * h) / 6.0
        )
        sy = (
            a * self.points[idx].y
            + b * self.points[idx + 1].y
            + ((a * a * a - a) * self._second_derivatives_y[idx]
            + (b * b * b - b) * self._second_derivatives_y[idx + 1])
            * (h * h) / 6.0
        )
        sz = (
            a * self.points[idx].z
            + b * self.points[idx + 1].z
            + ((a * a * a - a) * self._second_derivatives_z[idx]
            + (b * b * b - b) * self._second_derivatives_z[idx + 1])
            * (h * h) / 6.0
        )
        return Vector3(sx, sy, sz)


def sample_spline(spline: CubicSpline, times: Iterable[float]) -> list[Vector3]:
    """Convenience helper returning spline evaluations at each time."""

    return [spline.evaluate(t) for t in times]


__all__ = [
    "GRAVITY",
    "CAR_MAX_SPEED",
    "CAR_MAX_ACCELERATION",
    "BOOST_ACCELERATION",
    "BOOST_CONSUMPTION_PER_SECOND",
    "advance_linear",
    "apply_ground_bounce",
    "advance_ball",
    "advance_car",
    "estimate_time_to_cover",
    "estimate_required_speed",
    "predict_backboard_bounce",
    "CubicSpline",
    "sample_spline",
]

