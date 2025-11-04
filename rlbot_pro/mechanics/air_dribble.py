"""Air-dribble mechanic with touch planning and deterministic corrections."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import (
    Vector3,
    clamp,
    dot,
    magnitude,
    normalize,
    subtract,
)
from ..physics_helpers import estimate_required_speed
from ..state import GameState


@dataclass(frozen=True)
class AirDribbleParams:
    """Parameters describing the desired air dribble."""

    carry_offset: Vector3
    target_velocity: float
    alignment_threshold: float = 0.65
    lateral_gain: float = 0.75
    vertical_gain: float = 0.8
    derivative_gain: float = 0.15
    minimum_boost: float = 5.0


class AirDribbleMechanic:
    """Maintain control of the ball in the air through smooth corrections."""

    def __init__(self, params: AirDribbleParams) -> None:
        self.params = params
        self._prepared = False
        self._last_error = Vector3(0.0, 0.0, 0.0)
        self._complete = False
        self._invalid = False

    def prep(self, state: GameState) -> None:
        self._prepared = True
        self._last_error = self._compute_error(state)
        self._complete = False
        self._invalid = state.car.is_demolished

    def _compute_error(self, state: GameState) -> Vector3:
        desired_touch = subtract(state.ball.position, self.params.carry_offset)
        return subtract(desired_touch, state.car.position)

    def _forward(self, state: GameState) -> Vector3:
        forward = normalize(state.car.forward)
        if magnitude(forward) <= 1e-5:
            return Vector3(1.0, 0.0, 0.0)
        return forward

    def step(self, state: GameState, dt: float = 1.0 / 60.0) -> Controls:
        if not self._prepared:
            self.prep(state)
        if self._invalid:
            return Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        error = self._compute_error(state)
        error_rate = Vector3(
            (error.x - self._last_error.x) / max(dt, 1e-3),
            (error.y - self._last_error.y) / max(dt, 1e-3),
            (error.z - self._last_error.z) / max(dt, 1e-3),
        )
        self._last_error = error

        forward = self._forward(state)
        ball_direction = normalize(subtract(state.ball.position, state.car.position))
        alignment = dot(forward, ball_direction)

        throttle_speed = estimate_required_speed(magnitude(error), 0.8)
        throttle = clamp((throttle_speed - dot(state.car.velocity, forward)) / 600.0, 0.0, 1.0)
        steer = clamp(
            (error.y * self.params.lateral_gain) + error_rate.y * self.params.derivative_gain,
            -1.0,
            1.0,
        )
        pitch = clamp(
            (-error.z * self.params.vertical_gain) - error_rate.z * self.params.derivative_gain,
            -1.0,
            1.0,
        )

        boost = (
            alignment > self.params.alignment_threshold
            and state.car.boost > self.params.minimum_boost
            and self.params.target_velocity > 1300.0
        )

        jump = False
        if state.car.on_ground and state.car.has_jump:
            vertical_gap = state.ball.position.z - state.car.position.z
            if vertical_gap > 200.0 or abs(error.z) < 150.0:
                jump = True

        self._complete = (
            magnitude(error) < 120.0
            and state.ball.position.z > state.car.position.z + self.params.carry_offset.z * 0.5
        )
        self._invalid = state.car.boost <= 0.5 and alignment < 0.0

        return Controls(
            throttle=throttle,
            steer=steer,
            pitch=pitch,
            yaw=steer,
            roll=0.0,
            boost=boost,
            jump=jump,
            handbrake=False,
        )

    def is_complete(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._complete

    def is_invalid(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._invalid


__all__ = ["AirDribbleParams", "AirDribbleMechanic"]

