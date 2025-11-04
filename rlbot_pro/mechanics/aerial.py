from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rlbot_pro.control import Controls
from rlbot_pro.math3d import vec3, normalize, dot, angle_between, clamp
from rlbot_pro.physics_helpers import (
    GRAVITY,
    CAR_MAX_SPEED,
    BOOST_ACCELERATION,
    constant_acceleration_kinematics,
)
from rlbot_pro.sensing import (
    get_car_forward_vector,
    get_car_up_vector,
    predict_ball_pos_at_time,
)
from rlbot_pro.state import GameState, Vector
from rlbot_pro.mechanics import Mechanic, get_pitch_yaw_roll


@dataclass
class AerialParams:
    """Parameters for the Aerial mechanic."""

    target_pos: Vector = (0.0, 0.0, 0.0)
    target_time: float = 0.0
    # How much boost to use for alignment vs. speed
    alignment_boost_threshold: float = 0.8
    # Max angle off target to still consider boosting for alignment
    max_alignment_angle: float = np.deg2rad(15)


@dataclass
class Aerial(Mechanic):
    """
    Executes an aerial maneuver to intercept the ball.
    PD orientation to a moving intercept point; boost gating by alignment.
    """

    params: AerialParams = field(default_factory=AerialParams)
    _target_intercept_pos: Optional[np.ndarray] = None
    _target_intercept_time: Optional[float] = None
    _start_time: float = 0.0

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Initializes the aerial mechanic by predicting an intercept point.
        Requires 'target_pos' and 'target_time' in kwargs for initial target.
        """
        self._start_time = gs.car.time
        self.params.target_pos = kwargs.get("target_pos", self.params.target_pos)
        self.params.target_time = kwargs.get("target_time", self.params.target_time)

        # Predict ball position at target time
        predicted_ball_pos = predict_ball_pos_at_time(gs.ball, self.params.target_time)
        self._target_intercept_pos = predicted_ball_pos
        self._target_intercept_time = gs.car.time + self.params.target_time

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates controls to orient the car towards the intercept point and boost.
        """
        controls = Controls()
        car_pos = np.array(gs.car.pos)
        car_vel = np.array(gs.car.vel)
        car_forward = get_car_forward_vector(gs.car)
        car_up = get_car_up_vector(gs.car)

        if self._target_intercept_pos is None or self._target_intercept_time is None:
            # Fallback if prep wasn't called or failed
            self.prep(gs, **kwargs)
            if self._target_intercept_pos is None:
                return controls  # Still no target, return empty controls

        target_pos = self._target_intercept_pos
        time_remaining = self._target_intercept_time - gs.car.time

        if time_remaining <= 0.05:  # Very close to intercept time
            # Try to hit the ball
            target_direction = normalize(target_pos - car_pos)
            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_direction, vec3(0, 0, 1)
            )
            controls = Controls(
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
                boost=True,
            )
            return controls

        # Calculate required acceleration to reach target
        # s = v0*t + 0.5*a*t^2  =>  a = 2 * (s - v0*t) / t^2
        displacement = target_pos - car_pos
        required_accel = (
            2 * (displacement - car_vel * time_remaining) / (time_remaining**2 + 1e-6)
        )
        target_vel = car_vel + required_accel * time_remaining

        # Target direction is towards the required velocity
        target_direction = normalize(target_vel - car_vel)

        # PD controller for orientation
        pitch, yaw, roll = get_pitch_yaw_roll(
            car_forward, car_up, target_direction, vec3(0, 0, 1)
        )

        # Boost gating by alignment
        angle_to_target = angle_between(car_forward, target_direction)
        should_boost = (
            gs.car.boost > 0
            and angle_to_target < self.params.max_alignment_angle
            and np.linalg.norm(car_vel) < CAR_MAX_SPEED
        )

        controls = Controls(
            pitch=clamp(pitch * 3, -1.0, 1.0),  # Proportional control
            yaw=clamp(yaw * 3, -1.0, 1.0),
            roll=clamp(roll * 3, -1.0, 1.0),
            boost=should_boost,
            jump=False,  # Only jump to initiate aerial, not during
        )

        return controls

    def is_complete(self, gs: GameState) -> bool:
        """
        The aerial is complete if the car has passed the intercept point
        or if the ball is significantly far from the intercept point.
        """
        if self._target_intercept_pos is None or self._target_intercept_time is None:
            return True  # Invalid state, consider complete

        car_pos = np.array(gs.car.pos)
        ball_pos = np.array(gs.ball.pos)

        # Check if car has passed the intercept point (in the direction of travel)
        if dot(car_pos - self._target_intercept_pos, normalize(np.array(gs.car.vel))) > 0:
            return True

        # Check if ball is far from predicted intercept point
        predicted_ball_at_intercept = predict_ball_pos_at_time(
            gs.ball, self._target_intercept_time - gs.car.time
        )
        if np.linalg.norm(predicted_ball_at_intercept - self._target_intercept_pos) > 200:
            return True  # Ball deviated too much

        return False

    def is_invalid(self, gs: GameState) -> bool:
        """
        The aerial is invalid if the car runs out of boost,
        or if the target intercept time has passed significantly.
        """
        if self._target_intercept_time is None:
            return True

        # If car runs out of boost and is not already aligned
        if gs.car.boost == 0 and (self._target_intercept_time - gs.car.time) > 0.5:
            return True

        # If target time has passed and we haven't hit the ball
        if gs.car.time > self._target_intercept_time + 0.5:
            return True

        return False
