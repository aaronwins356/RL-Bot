"""Deterministic ceiling shot state machine."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, dot, magnitude, normalize, subtract
from ..state import GameState


@dataclass(frozen=True)
class CeilingShotParams:
    """Configuration for executing a ceiling shot."""

    carry_target: Vector3
    detach_height: float
    detach_time: float
    flip_window: tuple[float, float]
    approach_gain: float = 1.2


class CeilingShotMechanic:
    """Carry the ball up the wall, detach from the ceiling, then flip."""

    def __init__(self, params: CeilingShotParams) -> None:
        self.params = params
        self._prepared = False
        self._elapsed = 0.0
        self._phase = "carry"
        self._flip_issued = False
        self._detached_at = 0.0
        self._complete = False
        self._invalid = False

    def prep(self, state: GameState) -> None:
        self._prepared = True
        self._elapsed = 0.0
        self._phase = "carry"
        self._flip_issued = False
        self._detached_at = 0.0
        self._complete = False
        self._invalid = state.car.is_demolished

    def _approach_wall(self, state: GameState) -> Controls:
        to_target = subtract(self.params.carry_target, state.car.position)
        direction = normalize(to_target)
        forward = normalize(state.car.forward)
        pitch_error = direction.z - forward.z
        yaw_error = direction.y - forward.y
        throttle = clamp(magnitude(to_target) / 1500.0, 0.2, 1.0)
        pitch = clamp(-pitch_error * self.params.approach_gain, -1.0, 1.0)
        yaw = clamp(yaw_error * self.params.approach_gain, -1.0, 1.0)
        roll = 0.0
        jump = state.car.on_ground and to_target.z > 200.0 and state.car.has_jump
        boost = dot(forward, direction) > 0.7 and state.car.boost > 10.0
        return Controls(throttle, yaw, pitch, yaw, roll, boost, jump, False)

    def _detach(self, state: GameState) -> Controls:
        forward = normalize(state.car.forward)
        throttle = 0.2
        pitch = clamp(-forward.z * 1.5, -1.0, 1.0)
        yaw = 0.0
        roll = 0.0
        boost = False
        jump = False
        return Controls(throttle, yaw, pitch, yaw, roll, boost, jump, False)

    def _flip(self) -> Controls:
        self._flip_issued = True
        return Controls(throttle=0.0, steer=0.0, pitch=-1.0, yaw=0.0, roll=0.0, boost=False, jump=True, handbrake=False)

    def step(self, state: GameState, dt: float = 1.0 / 60.0) -> Controls:
        if not self._prepared:
            self.prep(state)
        if self._invalid:
            return Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        self._elapsed += dt
        controls = Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        if self._phase == "carry":
            controls = self._approach_wall(state)
            if state.car.position.z >= self.params.detach_height or self._elapsed >= self.params.detach_time:
                self._phase = "detach"
                self._detached_at = self._elapsed
        elif self._phase == "detach":
            controls = self._detach(state)
            if self._elapsed - self._detached_at >= 0.1:
                self._phase = "flip"
        elif self._phase == "flip":
            window_start, window_end = self.params.flip_window
            time_since_detach = self._elapsed - self._detached_at
            allowance = dt * 0.5
            if not self._flip_issued and window_start <= time_since_detach <= window_end + allowance:
                controls = self._flip()
            elif time_since_detach > window_end + allowance and not self._flip_issued:
                self._invalid = True
            else:
                controls = Controls(0.3, 0.0, -0.3, 0.0, 0.0, False, False, False)
            if self._flip_issued:
                self._phase = "complete"
        else:
            controls = Controls(0.4, 0.0, -0.2, 0.0, 0.0, False, False, False)

        self._complete = self._phase == "complete"
        if state.car.on_ground and self._phase != "carry":
            self._invalid = True

        return controls

    def is_complete(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._complete

    def is_invalid(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._invalid


__all__ = ["CeilingShotParams", "CeilingShotMechanic"]

