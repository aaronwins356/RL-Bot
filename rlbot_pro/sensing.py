"""Heuristic sensing utilities."""

from __future__ import annotations

from math import isclose

from .math3d import Vector3, dot, magnitude, subtract
from .state import GameState


def time_to_ball_touch(state: GameState, minimum_speed: float = 50.0) -> float:
    """Estimate seconds until the car reaches the ball."""
    relative = subtract(state.ball.position, state.car.position)
    distance = magnitude(relative)
    relative_velocity = subtract(state.ball.velocity, state.car.velocity)
    directional_speed = dot(relative_velocity, relative) / (distance + 1e-6)
    speed_toward_ball = max(minimum_speed, abs(directional_speed) + minimum_speed)
    return distance / speed_toward_ball


def is_ball_heading_to_goal(state: GameState, goal_direction: Vector3) -> bool:
    """Return True if the ball velocity is aligned with the goal direction."""
    velocity_mag = magnitude(state.ball.velocity)
    goal_mag = magnitude(goal_direction)
    if isclose(velocity_mag, 0.0, abs_tol=1e-6) or isclose(goal_mag, 0.0, abs_tol=1e-6):
        return False
    alignment = dot(state.ball.velocity, goal_direction) / (velocity_mag * goal_mag)
    return alignment > 0.5


__all__ = ["time_to_ball_touch", "is_ball_heading_to_goal"]
