"""Flip reset planning with deterministic contact tracking."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, dot, magnitude, normalize, subtract
from ..state import GameState


@dataclass(frozen=True)
class FlipResetParams:
    """Parameterization of a flip reset attempt."""

    target_surface_normal: Vector3
    commit_time: float
    max_resets: int = 1
    touch_tolerance: float = 120.0
    flip_cooldown: float = 0.2


class FlipResetMechanic:
    """Track underside contact and trigger flips after resets."""

    def __init__(self, params: FlipResetParams) -> None:
        self.params = params
        self._prepared = False
        self._time_remaining = params.commit_time
        self._resets = 0
        self._flip_timer = 0.0
        self._complete = False
        self._invalid = False

    def prep(self, state: GameState) -> None:
        self._prepared = True
        self._time_remaining = self.params.commit_time
        self._resets = 0
        self._flip_timer = 0.0
        self._complete = False
        self._invalid = state.car.is_demolished

    def _direction_to_ball(self, state: GameState) -> Vector3:
        return normalize(subtract(state.ball.position, state.car.position))

    def _underside_contact(self, state: GameState) -> bool:
        direction = self._direction_to_ball(state)
        separation = magnitude(subtract(state.ball.position, state.car.position))
        underside_alignment = dot(state.car.up, direction)
        return separation < self.params.touch_tolerance and underside_alignment < -0.2

    def step(self, state: GameState, dt: float = 1.0 / 60.0) -> Controls:
        if not self._prepared:
            self.prep(state)
        if self._invalid:
            return Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        self._time_remaining = max(0.0, self._time_remaining - dt)
        direction = self._direction_to_ball(state)
        forward = normalize(state.car.forward)
        pitch_error = direction.z - forward.z
        yaw_error = direction.y - forward.y

        pitch = clamp(-pitch_error * 1.5, -1.0, 1.0)
        yaw = clamp(yaw_error * 1.2, -1.0, 1.0)
        surface_alignment = dot(state.car.up, self.params.target_surface_normal)
        roll = clamp(-surface_alignment * 0.5, -1.0, 1.0)
        throttle = 0.8
        boost = dot(forward, direction) > 0.5 and state.car.boost > 0.0

        jump = False
        if self._time_remaining <= 0.0 and state.car.has_jump and not state.car.on_ground:
            jump = True

        if self._underside_contact(state):
            self._resets = min(self._resets + 1, self.params.max_resets)
            self._flip_timer = self.params.flip_cooldown

        if self._flip_timer > 0.0:
            self._flip_timer = max(0.0, self._flip_timer - dt)
            if self._flip_timer == 0.0 and self._resets > 0 and state.car.has_jump:
                jump = True
                self._resets -= 1

        self._complete = self._resets == 0 and self._time_remaining == 0.0 and not state.car.has_jump
        self._invalid = state.car.on_ground or state.car.is_demolished

        return Controls(throttle, yaw, pitch, yaw, roll, boost, jump, False)

    def is_complete(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._complete

    def is_invalid(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._invalid


__all__ = ["FlipResetParams", "FlipResetMechanic"]

