"""Deterministic aerial controller with PD orientation and boost gating."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, dot, magnitude, normalize, subtract
from ..physics_helpers import estimate_required_speed
from ..state import GameState


@dataclass(frozen=True)
class AerialParams:
    """Parameters for guiding an aerial intercept."""

    intercept: Vector3
    arrival_time: float
    orientation_kp: float = 2.4
    orientation_kd: float = 0.35
    boost_alignment_threshold: float = 0.6
    minimum_distance: float = 85.0


class AerialMechanic:
    """PD-based aerial controller that converges on a target point."""

    def __init__(self, params: AerialParams) -> None:
        self.params = params
        self._prepared = False
        self._jump_issued = False
        self._target_direction = Vector3(0.0, 0.0, 1.0)
        self._time_remaining = params.arrival_time
        self._complete = False
        self._invalid = False

    def prep(self, state: GameState) -> None:
        car_to_target = subtract(self.params.intercept, state.car.position)
        distance = magnitude(car_to_target)
        self._target_direction = normalize(car_to_target)
        self._time_remaining = self.params.arrival_time
        self._jump_issued = not state.car.on_ground
        self._prepared = True
        self._complete = distance <= self.params.minimum_distance
        self._invalid = state.car.is_demolished

    def _forward(self, state: GameState) -> Vector3:
        forward = normalize(state.car.forward)
        if magnitude(forward) <= 1e-5:
            velocity_dir = normalize(state.car.velocity)
            if magnitude(velocity_dir) <= 1e-5:
                return Vector3(1.0, 0.0, 0.0)
            return velocity_dir
        return forward

    def step(self, state: GameState, dt: float = 1.0 / 60.0) -> Controls:
        if not self._prepared:
            self.prep(state)
        if self._invalid:
            return Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        car_to_target = subtract(self.params.intercept, state.car.position)
        distance = magnitude(car_to_target)
        direction = normalize(car_to_target)
        forward = self._forward(state)
        alignment = dot(forward, direction)

        pitch_error = direction.z - forward.z
        yaw_error = direction.y - forward.y
        roll_error = direction.x - forward.x

        pitch = clamp(
            -pitch_error * self.params.orientation_kp
            - state.car.angular_velocity.y * self.params.orientation_kd,
            -1.0,
            1.0,
        )
        yaw = clamp(
            yaw_error * self.params.orientation_kp
            - state.car.angular_velocity.z * self.params.orientation_kd,
            -1.0,
            1.0,
        )
        roll = clamp(
            -roll_error * (self.params.orientation_kp * 0.5)
            - state.car.angular_velocity.x * self.params.orientation_kd,
            -1.0,
            1.0,
        )

        relative_speed = dot(state.car.velocity, direction)
        required_speed = estimate_required_speed(distance, max(self._time_remaining, 1e-2))
        throttle = clamp((required_speed - relative_speed) / 500.0, 0.0, 1.0)

        boost = (
            alignment >= self.params.boost_alignment_threshold
            and state.car.boost > 1.0
            and distance > self.params.minimum_distance
        )

        jump = False
        if state.car.on_ground and not self._jump_issued and distance > 120.0:
            jump = True
            self._jump_issued = True

        self._time_remaining = max(0.0, self._time_remaining - dt)
        self._complete = distance <= self.params.minimum_distance and alignment > 0.92
        self._invalid = (
            state.car.is_demolished
            or (self._time_remaining <= 0.0 and distance > self.params.minimum_distance * 2.0)
        )

        return Controls(
            throttle=throttle,
            steer=yaw,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
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


__all__ = ["AerialParams", "AerialMechanic"]

