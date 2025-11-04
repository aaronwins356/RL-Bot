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
class AirDribbleParams:
    """Parameters for the Air Dribble mechanic."""

    # Desired offset from the ball (e.g., slightly behind and below)
    offset: Vector = (0.0, -100.0, -50.0)
    # Desired ball velocity to maintain
    target_ball_vel: Vector = (0.0, 1000.0, 0.0)
    # How close to the ball before making a touch
    touch_distance: float = 100.0
    # Max angle off target to still consider boosting for alignment
    max_alignment_angle: float = np.deg2rad(10)


@dataclass
class AirDribble(Mechanic):
    """
    Executes an air dribble maneuver.
    Maintain offset and target ball velocity; small impulse touches.
    """

    params: AirDribbleParams = field(default_factory=AirDribbleParams)
    _is_dribbling: bool = False
    _last_touch_time: float = 0.0
    _dribble_start_time: float = 0.0

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Initializes the air dribble mechanic.
        """
        self._dribble_start_time = gs.car.time
        self._is_dribbling = False
        self._last_touch_time = 0.0

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates controls to maintain position relative to the ball and make touches.
        """
        controls = Controls()
        car_pos = np.array(gs.car.pos)
        car_vel = np.array(gs.car.vel)
        car_forward = get_car_forward_vector(gs.car)
        car_up = get_car_up_vector(gs.car)
        ball_pos = np.array(gs.ball.pos)
        ball_vel = np.array(gs.ball.vel)

        # Calculate desired car position relative to the ball
        # Rotate offset by ball's current direction of travel (simplified)
        # For a true air dribble, this would be more complex, considering car orientation
        # For now, assume target_ball_vel defines the "forward" for the dribble
        dribble_forward = normalize(np.array(self.params.target_ball_vel))
        # Simple rotation of offset to align with dribble_forward
        # This is a very basic approximation and would need a proper rotation matrix
        # if the offset was not primarily along one axis.
        # For (0, -100, -50) offset, if dribble_forward is (0,1,0), no change.
        # If dribble_forward is (1,0,0), offset becomes (-100, 0, -50)
        # A more robust solution would involve a rotation matrix from (0,1,0) to dribble_forward
        
        # For simplicity, let's just add the offset directly to the ball's position
        # and let the orientation logic handle the rest.
        target_car_pos = ball_pos + np.array(self.params.offset)

        # Calculate target car velocity to match ball velocity
        target_car_vel = np.array(self.params.target_ball_vel)

        # Calculate required acceleration to reach target car velocity
        # a = (v_target - v_current) / dt
        accel_needed = (target_car_vel - car_vel) / (gs.dt + 1e-6)
        
        # Target direction for car orientation
        # Aim slightly ahead of the ball to push it, or towards target_car_pos
        target_direction = normalize(target_car_pos - car_pos)

        pitch, yaw, roll = get_pitch_yaw_roll(
            car_forward, car_up, target_direction, vec3(0, 0, 1)
        )

        controls = Controls(
            pitch=clamp(pitch * 3, -1.0, 1.0),
            yaw=clamp(yaw * 3, -1.0, 1.0),
            roll=clamp(roll * 3, -1.0, 1.0),
            throttle=1.0, # Always try to maintain speed
            boost=False,
        )

        # Boost to match target velocity and maintain position
        # Only boost if we are behind the ball and need to catch up or push
        if dot(car_forward, normalize(ball_pos - car_pos)) > 0.5: # Car is generally facing the ball
            if np.linalg.norm(car_vel) < np.linalg.norm(target_car_vel) * 0.95:
                controls = controls._replace(boost=True)
            
            # Small impulse touches when close enough
            dist_to_ball = np.linalg.norm(ball_pos - car_pos)
            if dist_to_ball < self.params.touch_distance and (gs.car.time - self._last_touch_time > 0.1):
                # Make a small jump to impart an impulse
                controls = controls._replace(jump=True)
                self._last_touch_time = gs.car.time
                self._is_dribbling = True # Indicate we are actively dribbling

        return controls

    def is_complete(self, gs: GameState) -> bool:
        """
        The air dribble is complete if the ball is too far from the car,
        or if the ball has landed.
        """
        if not self._is_dribbling:
            return False # Not yet started dribbling

        car_pos = np.array(gs.car.pos)
        ball_pos = np.array(gs.ball.pos)

        # If ball is on the ground
        if ball_pos[2] < 100: # Ball height threshold
            return True
        
        # If car is too far from the ball
        if np.linalg.norm(ball_pos - car_pos) > 500: # Distance threshold
            return True

        return False

    def is_invalid(self, gs: GameState) -> bool:
        """
        The air dribble is invalid if the car is on the ground,
        or if the ball is too far away to initiate/continue.
        """
        if gs.car.on_ground:
            return True
        
        car_pos = np.array(gs.car.pos)
        ball_pos = np.array(gs.ball.pos)

        # If ball is too far to even start a dribble
        if np.linalg.norm(ball_pos - car_pos) > 1000 and not self._is_dribbling:
            return True

        return False
