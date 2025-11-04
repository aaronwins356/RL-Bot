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
    estimate_ball_bounce,
)
from rlbot_pro.sensing import (
    get_car_forward_vector,
    get_car_up_vector,
    predict_ball_pos_at_time,
    get_car_local_coords,
)
from rlbot_pro.state import GameState, Vector
from rlbot_pro.mechanics import Mechanic, get_pitch_yaw_roll


@dataclass
class DoubleTapParams:
    """Parameters for the Double Tap mechanic."""

    # Max time to wait for a backboard bounce
    max_bounce_time: float = 2.0
    # Desired target for the first touch (e.g., towards opponent's net)
    first_touch_target: Vector = (0.0, 5120.0, 0.0)
    # How close to the ball for the first touch
    first_touch_distance: float = 150.0
    # How close to the ball for the second touch
    second_touch_distance: float = 150.0


@dataclass
class DoubleTap(Mechanic):
    """
    Executes a double tap maneuver.
    Backboard read: predict bounce point, plan first touch target, then chase second intercept.
    """

    params: DoubleTapParams = field(default_factory=DoubleTapParams)
    _state: str = "INIT"  # INIT, APPROACH_FIRST_TOUCH, FIRST_TOUCH_MADE, CHASE_SECOND_TOUCH
    _bounce_pos: Optional[np.ndarray] = None
    _bounce_time: Optional[float] = None
    _first_touch_pos: Optional[np.ndarray] = None
    _first_touch_time: Optional[float] = None
    _second_touch_target_pos: Optional[np.ndarray] = None

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Initializes the double tap mechanic by predicting a backboard bounce.
        """
        self._state = "INIT"
        self._bounce_pos = None
        self._bounce_time = None
        self._first_touch_pos = None
        self._first_touch_time = None
        self._second_touch_target_pos = None

        # Estimate ball bounce on the backboard
        bounce_result = estimate_ball_bounce(
            gs.ball, np.array(gs.car.pos), max_time=self.params.max_bounce_time
        )
        if bounce_result:
            self._bounce_pos, self._bounce_time = bounce_result
            # Plan first touch target based on bounce
            # Aim to hit the ball towards the center of the opponent's net after the bounce
            self._first_touch_pos = self._bounce_pos + normalize(
                np.array(self.params.first_touch_target) - self._bounce_pos
            ) * 100
            self._first_touch_time = gs.car.time + self._bounce_time

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates controls based on the current state of the double tap.
        """
        controls = Controls()
        car_pos = np.array(gs.car.pos)
        car_vel = np.array(gs.car.vel)
        car_forward = get_car_forward_vector(gs.car)
        car_up = get_car_up_vector(gs.car)
        ball_pos = np.array(gs.ball.pos)
        ball_vel = np.array(gs.ball.vel)

        if self._state == "INIT":
            if self._first_touch_pos is not None:
                self._state = "APPROACH_FIRST_TOUCH"
            else:
                # No viable bounce, just drive towards ball
                target_direction = normalize(ball_pos - car_pos)
                pitch, yaw, roll = get_pitch_yaw_roll(
                    car_forward, car_up, target_direction, vec3(0, 0, 1)
                )
                return Controls(
                    throttle=1.0,
                    boost=True,
                    pitch=clamp(pitch * 3, -1.0, 1.0),
                    yaw=clamp(yaw * 3, -1.0, 1.0),
                    roll=clamp(roll * 3, -1.0, 1.0),
                )

        if self._state == "APPROACH_FIRST_TOUCH":
            if self._first_touch_pos is None:
                self._state = "INIT"  # Re-evaluate if target disappeared
                return controls

            target_direction = normalize(self._first_touch_pos - car_pos)
            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_direction, vec3(0, 0, 1)
            )
            controls = Controls(
                throttle=1.0,
                boost=True,
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
            )

            if np.linalg.norm(ball_pos - car_pos) < self.params.first_touch_distance:
                self._state = "FIRST_TOUCH_MADE"
                # Predict second touch target after first touch
                # This is a very simplified prediction; a real bot would use ball prediction
                self._second_touch_target_pos = predict_ball_pos_at_time(gs.ball, 0.5) # Predict 0.5s after first touch

        elif self._state == "FIRST_TOUCH_MADE":
            # Chase the ball for the second touch
            if self._second_touch_target_pos is None:
                # Fallback to just chasing the ball
                target_direction = normalize(ball_pos - car_pos)
            else:
                target_direction = normalize(self._second_touch_target_pos - car_pos)

            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_direction, vec3(0, 0, 1)
            )
            controls = Controls(
                throttle=1.0,
                boost=True,
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
            )

            if np.linalg.norm(ball_pos - car_pos) < self.params.second_touch_distance:
                self._state = "CHASE_SECOND_TOUCH" # Indicate we are ready for second touch

        elif self._state == "CHASE_SECOND_TOUCH":
            # Attempt the second touch
            target_direction = normalize(ball_pos - car_pos)
            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_direction, vec3(0, 0, 1)
            )
            controls = Controls(
                throttle=1.0,
                boost=True,
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
                jump=True, # Attempt a jump to hit the ball
            )

        return controls

    def is_complete(self, gs: GameState) -> bool:
        """
        The double tap is complete if the ball has been hit twice,
        or if the ball is too far from the car.
        """
        # Very simplified: assume complete if ball has high velocity after first touch
        if self._state == "CHASE_SECOND_TOUCH" and np.linalg.norm(np.array(gs.ball.vel)) > 1500:
            return True
        
        # If ball is too far from car after first touch
        if self._state != "INIT" and np.linalg.norm(np.array(gs.ball.pos) - np.array(gs.car.pos)) > 1000:
            return True

        return False

    def is_invalid(self, gs: GameState) -> bool:
        """
        The double tap is invalid if the initial bounce prediction fails,
        or if the car is too far from the ball at any stage.
        """
        if self._state == "INIT" and self._bounce_pos is None:
            return True
        
        # If car is on ground during aerial phase of double tap
        if self._state in ["FIRST_TOUCH_MADE", "CHASE_SECOND_TOUCH"] and gs.car.on_ground:
            return True
        
        # If ball is too far to initiate
        if self._state == "INIT" and np.linalg.norm(np.array(gs.ball.pos) - np.array(gs.car.pos)) > 2000:
            return True

        return False
