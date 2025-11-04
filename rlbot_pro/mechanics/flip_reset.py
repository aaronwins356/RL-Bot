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
    get_car_local_coords,
)
from rlbot_pro.state import GameState, Vector
from rlbot_pro.mechanics import Mechanic, get_pitch_yaw_roll


@dataclass
class FlipResetParams:
    """Parameters for the Flip Reset mechanic."""

    # How close to the ball to attempt a flip reset
    contact_distance: float = 100.0
    # Time window after contact to use the flip
    flip_window: float = 0.2
    # Desired car orientation for reset (e.g., upside down)
    desired_up_vector: Vector = (0.0, 0.0, -1.0)


@dataclass
class FlipReset(Mechanic):
    """
    Executes a flip reset maneuver.
    Underside contact heuristic; timed flip usage; reset state.
    """

    params: FlipResetParams = field(default_factory=FlipResetParams)
    _state: str = "INIT"  # INIT, APPROACHING, CONTACT, FLIP_USED, COMPLETE
    _contact_time: float = 0.0
    _has_reset_flip: bool = False

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Initializes the flip reset mechanic.
        """
        self._state = "INIT"
        self._contact_time = 0.0
        self._has_reset_flip = False

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates controls to approach the ball, make contact, and use the flip.
        """
        controls = Controls()
        car_pos = np.array(gs.car.pos)
        car_vel = np.array(gs.car.vel)
        car_forward = get_car_forward_vector(gs.car)
        car_up = get_car_up_vector(gs.car)
        ball_pos = np.array(gs.ball.pos)
        ball_vel = np.array(gs.ball.vel)

        dist_to_ball = np.linalg.norm(ball_pos - car_pos)

        if self._state == "INIT" or self._state == "APPROACHING":
            # Drive towards the ball, trying to get underneath it
            target_pos = ball_pos + vec3(0, 0, -100) # Aim slightly below the ball
            target_direction = normalize(target_pos - car_pos)
            
            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_direction, np.array(self.params.desired_up_vector)
            )

            controls = Controls(
                throttle=1.0,
                boost=True,
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
            )

            if dist_to_ball < self.params.contact_distance:
                self._state = "CONTACT"
                self._contact_time = gs.car.time

        elif self._state == "CONTACT":
            # After contact, try to get the flip reset
            # Check for underside contact heuristic: car's up vector pointing down
            # and car is close to ball
            if dot(car_up, vec3(0, 0, -1)) > 0.8 and dist_to_ball < self.params.contact_distance * 1.5:
                self._has_reset_flip = True
                self._state = "FLIP_USED" # Assume flip is used immediately after reset

            # Continue orienting for potential flip
            target_direction = normalize(ball_pos - car_pos)
            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_direction, np.array(self.params.desired_up_vector)
            )
            controls = Controls(
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
                boost=True,
            )

        elif self._state == "FLIP_USED":
            # After flip reset, use the flip if available and within window
            if self._has_reset_flip and (gs.car.time - self._contact_time) < self.params.flip_window:
                controls = Controls(jump=True) # Use the flip
                self._has_reset_flip = False # Flip used
                self._state = "COMPLETE" # Consider complete after flip

            # Continue orienting towards ball
            target_direction = normalize(ball_pos - car_pos)
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

    def is_complete(self, gs: GameState) -> bool:
        """
        The flip reset is complete if the flip has been used,
        or if the car is too far from the ball.
        """
        if self._state == "COMPLETE":
            return True
        
        # If car is too far from ball after attempting reset
        if self._state != "INIT" and np.linalg.norm(np.array(gs.ball.pos) - np.array(gs.car.pos)) > 500:
            return True

        return False

    def is_invalid(self, gs: GameState) -> bool:
        """
        The flip reset is invalid if the car is on the ground,
        or if the ball is too far away to attempt a reset.
        """
        if gs.car.on_ground:
            return True
        
        # If ball is too far to even start
        if self._state == "INIT" and np.linalg.norm(np.array(gs.ball.pos) - np.array(gs.car.pos)) > 1000:
            return True

        return False
