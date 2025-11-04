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
    is_car_on_ceiling,
    is_car_on_wall,
)
from rlbot_pro.state import GameState, Vector
from rlbot_pro.mechanics import Mechanic, get_pitch_yaw_roll


@dataclass
class CeilingParams:
    """Parameters for the Ceiling mechanic."""

    # Desired time to detach from the ceiling after wall carry
    detach_time_offset: float = 0.5
    # Target position for the first touch after detaching
    first_touch_target: Vector = (0.0, 0.0, 0.0)
    # How close to the ball to consider it a "carry" on the wall
    wall_carry_distance: float = 200.0


@dataclass
class Ceiling(Mechanic):
    """
    Executes a ceiling shot setup.
    Wall carry -> ceiling detach with timing param -> handoff point.
    """

    params: CeilingParams = field(default_factory=CeilingParams)
    _state: str = "INIT"  # INIT, WALL_CARRY, CEILING_DETACH, FIRST_TOUCH
    _wall_carry_start_time: float = 0.0
    _detach_target_time: float = 0.0
    _first_touch_target_pos: Optional[np.ndarray] = None

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Initializes the ceiling shot mechanic.
        """
        self._state = "INIT"
        self._wall_carry_start_time = 0.0
        self._detach_target_time = 0.0
        self._first_touch_target_pos = None
        self.params.first_touch_target = kwargs.get(
            "first_touch_target", self.params.first_touch_target
        )

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates controls based on the current state of the ceiling shot.
        """
        controls = Controls()
        car_pos = np.array(gs.car.pos)
        car_vel = np.array(gs.car.vel)
        car_forward = get_car_forward_vector(gs.car)
        car_up = get_car_up_vector(gs.car)
        ball_pos = np.array(gs.ball.pos)
        ball_vel = np.array(gs.ball.vel)

        if self._state == "INIT":
            # Look for opportunities to start a wall carry
            if is_car_on_wall(gs.car) and np.linalg.norm(ball_pos - car_pos) < self.params.wall_carry_distance:
                self._state = "WALL_CARRY"
                self._wall_carry_start_time = gs.car.time
                self._first_touch_target_pos = np.array(self.params.first_touch_target)
                self._detach_target_time = gs.car.time + self.params.detach_time_offset
                # Drive up the wall towards the ball
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
                )
            else:
                # If not ready, just drive towards the ball
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
                )

        elif self._state == "WALL_CARRY":
            # Maintain wall carry until detach time
            if gs.car.time < self._detach_target_time:
                # Continue driving up the wall, maintaining proximity to ball
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
                )
            else:
                # Time to detach
                self._state = "CEILING_DETACH"
                # Jump off the ceiling
                controls = Controls(jump=True)

        elif self._state == "CEILING_DETACH":
            # After detaching, orient towards the first touch target
            if not gs.car.on_ground: # Still in air
                if self._first_touch_target_pos is not None:
                    target_direction = normalize(self._first_touch_target_pos - car_pos)
                    pitch, yaw, roll = get_pitch_yaw_roll(
                        car_forward, car_up, target_direction, vec3(0, 0, 1)
                    )
                    controls = Controls(
                        pitch=clamp(pitch * 3, -1.0, 1.0),
                        yaw=clamp(yaw * 3, -1.0, 1.0),
                        roll=clamp(roll * 3, -1.0, 1.0),
                        boost=True,
                    )
                else:
                    # Fallback: just try to recover
                    controls = Controls(pitch=1.0) # Nose down to recover
            else:
                # Landed, transition to first touch
                self._state = "FIRST_TOUCH"

        elif self._state == "FIRST_TOUCH":
            # Drive towards the first touch target
            if self._first_touch_target_pos is not None:
                target_direction = normalize(self._first_touch_target_pos - car_pos)
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
            else:
                # Fallback: just drive towards ball
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
                )

        return controls

    def is_complete(self, gs: GameState) -> bool:
        """
        The ceiling shot is complete if the car has made the first touch
        or if the ball is too far away.
        """
        if self._state == "FIRST_TOUCH":
            # Check if car is close to the first touch target
            if self._first_touch_target_pos is not None:
                if np.linalg.norm(np.array(gs.car.pos) - self._first_touch_target_pos) < 150:
                    return True
            # Also check if ball is hit
            if np.linalg.norm(np.array(gs.ball.vel)) > 1000: # Ball has been hit hard
                return True
        
        # If ball is too far from car after initial setup
        if np.linalg.norm(np.array(gs.ball.pos) - np.array(gs.car.pos)) > 1500 and self._state != "INIT":
            return True

        return False

    def is_invalid(self, gs: GameState) -> bool:
        """
        The ceiling shot is invalid if the car loses the ball during wall carry,
        or if the car is not on the wall/ceiling when it should be.
        """
        if self._state == "WALL_CARRY" and not is_car_on_wall(gs.car):
            return True
        if self._state == "CEILING_DETACH" and gs.car.on_ground:
            return True # Should be in air after detach
        
        # If ball gets too far during wall carry
        if self._state == "WALL_CARRY" and np.linalg.norm(np.array(gs.ball.pos) - np.array(gs.car.pos)) > self.params.wall_carry_distance * 2:
            return True

        return False
