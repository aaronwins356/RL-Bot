"""Deterministic double-tap execution with bounce prediction."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, dot, magnitude, normalize, subtract, add, scale
from ..physics_helpers import estimate_required_speed, predict_backboard_bounce
from ..state import GameState


@dataclass(frozen=True)
class DoubleTapParams:
    """Configuration for a backboard double tap."""

    backboard_y: float
    restitution: float
    first_touch_speed: float
    second_arrival_time: float
    boost_alignment_threshold: float = 0.6


class DoubleTapMechanic:
    """Two-phase double tap using backboard bounce prediction."""

    def __init__(self, params: DoubleTapParams) -> None:
        self.params = params
        self._prepared = False
        self._bounce_point: Vector3 | None = None
        self._post_bounce_velocity: Vector3 | None = None
        self._bounce_time: float = 0.0
        self._phase = "setup"
        self._complete = False
        self._invalid = False

    def prep(self, state: GameState) -> None:
        impact, velocity, time_to_board = predict_backboard_bounce(
            state.ball.position, state.ball.velocity, self.params.backboard_y, self.params.restitution
        )
        self._bounce_point = impact
        self._post_bounce_velocity = velocity
        self._bounce_time = time_to_board
        self._phase = "setup"
        self._complete = False
        self._invalid = time_to_board == float("inf") or time_to_board <= 0.0 or state.car.is_demolished
        self._prepared = True

    def _forward(self, state: GameState) -> Vector3:
        forward = normalize(state.car.forward)
        if magnitude(forward) <= 1e-5:
            return Vector3(1.0, 0.0, 0.0)
        return forward

    def _approach_first_touch(self, state: GameState) -> Controls:
        assert self._bounce_point is not None
        intercept_target = Vector3(
            self._bounce_point.x,
            self._bounce_point.y - 150.0,
            max(self._bounce_point.z - 150.0, 500.0),
        )
        to_target = subtract(intercept_target, state.car.position)
        direction = normalize(to_target)
        forward = self._forward(state)
        throttle = clamp(self.params.first_touch_speed / 2300.0, 0.0, 1.0)
        steer = clamp(direction.y * 1.5, -1.0, 1.0)
        pitch = clamp(-direction.z * 1.2, -1.0, 1.0)
        boost = dot(forward, direction) > self.params.boost_alignment_threshold and state.car.boost > 10.0
        jump = state.car.on_ground and state.car.has_jump and self._bounce_time < 1.0
        return Controls(throttle, steer, pitch, steer, 0.0, boost, jump, False)

    def _second_intercept(self, state: GameState) -> Controls:
        assert self._bounce_point is not None
        assert self._post_bounce_velocity is not None
        target_time = self.params.second_arrival_time
        travel = subtract(
            add(self._bounce_point, scale(self._post_bounce_velocity, target_time)),
            state.car.position,
        )
        direction = normalize(travel)
        forward = self._forward(state)
        distance = magnitude(travel)
        required_speed = estimate_required_speed(distance, max(target_time, 0.2))
        throttle = clamp(required_speed / 2300.0, 0.0, 1.0)
        steer = clamp(direction.y * 1.8, -1.0, 1.0)
        pitch = clamp(-direction.z * 1.5, -1.0, 1.0)
        boost = dot(forward, direction) > self.params.boost_alignment_threshold and state.car.boost > 0.0
        jump = not state.car.on_ground and state.car.has_jump and distance < 250.0
        return Controls(throttle, steer, pitch, steer, 0.0, boost, jump, False)

    def step(self, state: GameState, dt: float = 1.0 / 60.0) -> Controls:  # pylint: disable=unused-argument
        if not self._prepared:
            self.prep(state)
        if self._invalid or self._bounce_point is None:
            return Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        if self._phase == "setup":
            controls = self._approach_first_touch(state)
            if state.ball.position.y >= self.params.backboard_y - 50.0:
                self._phase = "finish"
            return controls

        controls = self._second_intercept(state)
        distance_to_target = magnitude(subtract(self._bounce_point, state.car.position))
        self._complete = distance_to_target < 150.0 and not state.car.has_jump
        self._invalid = state.car.boost <= 0.0 and distance_to_target > 600.0
        return controls

    def is_complete(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._complete

    def is_invalid(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._invalid


__all__ = ["DoubleTapParams", "DoubleTapMechanic"]

