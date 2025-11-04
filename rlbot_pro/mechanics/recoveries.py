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
    get_car_right_vector,
    get_car_angular_velocity_local,
)
from rlbot_pro.state import GameState, Vector
from rlbot_pro.mechanics import Mechanic, get_pitch_yaw_roll


@dataclass
class RecoveriesParams:
    """Parameters for the Recoveries mechanic."""

    # Threshold for car's Z-axis up vector dot product to determine if upside down
    upside_down_threshold: float = -0.5
    # Threshold for car's Z-axis up vector dot product to determine if on side
    on_side_threshold: float = 0.5
    # Minimum speed to attempt a wavedash
    wavedash_min_speed: float = 500.0
    # Time window to perform a half-flip after jumping
    half_flip_window: float = 0.15


@dataclass
class Recoveries(Mechanic):
    """
    Executes various recovery maneuvers: wavedash, half-flip, upright reorientation.
    Chooses the cheapest recovery based on car state.
    """

    params: RecoveriesParams = field(default_factory=RecoveriesParams)
    _state: str = "INIT"  # INIT, WAVEDASH, HALF_FLIP_JUMP, HALF_FLIP_PITCH, UPRIGHT
    _jump_time: float = 0.0
    _initial_car_up: Optional[np.ndarray] = None

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Initializes the recovery mechanic.
        """
        self._state = "INIT"
        self._jump_time = 0.0
        self._initial_car_up = None

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates controls for the appropriate recovery maneuver.
        """
        controls = Controls()
        car_pos = np.array(gs.car.pos)
        car_vel = np.array(gs.car.vel)
        car_forward = get_car_forward_vector(gs.car)
        car_up = get_car_up_vector(gs.car)
        car_right = get_car_right_vector(gs.car)
        local_ang_vel = get_car_angular_velocity_local(gs.car)

        # Determine current car orientation relative to ground
        up_dot_world_up = dot(car_up, vec3(0, 0, 1))

        if gs.car.on_ground:
            self._state = "COMPLETE" # If on ground, recovery is complete
            return controls

        if self._state == "INIT":
            if up_dot_world_up < self.params.upside_down_threshold:
                # Car is upside down, consider half-flip
                self._state = "HALF_FLIP_JUMP"
                self._jump_time = gs.car.time
                self._initial_car_up = car_up
            elif abs(up_dot_world_up) < self.params.on_side_threshold:
                # Car is on its side, consider upright reorientation
                self._state = "UPRIGHT"
            else:
                # Default to upright if not clearly upside down or on side
                self._state = "UPRIGHT"

        if self._state == "HALF_FLIP_JUMP":
            # First jump of a half-flip
            controls = Controls(jump=True)
            if gs.car.time - self._jump_time > 0.05: # Small delay after jump
                self._state = "HALF_FLIP_PITCH"

        elif self._state == "HALF_FLIP_PITCH":
            # Pitch back and roll for half-flip
            controls = Controls(
                pitch=-1.0,  # Pitch back
                roll=1.0 if dot(car_right, vec3(0, 0, 1)) > 0 else -1.0, # Roll to get upright
                jump=False, # Don't hold jump
            )
            # If car is mostly upright, consider it done
            if up_dot_world_up > 0.8:
                self._state = "COMPLETE"

        elif self._state == "UPRIGHT":
            # Reorient car to be upright
            # Target up vector is world up (0,0,1)
            # Target forward vector can be current forward to maintain direction
            target_forward = car_forward
            target_up = vec3(0, 0, 1)

            pitch, yaw, roll = get_pitch_yaw_roll(
                car_forward, car_up, target_forward, target_up
            )

            controls = Controls(
                pitch=clamp(pitch * 3, -1.0, 1.0),
                yaw=clamp(yaw * 3, -1.0, 1.0),
                roll=clamp(roll * 3, -1.0, 1.0),
                boost=True, # Boost to gain control faster
            )
            # If car is mostly upright and stable
            if up_dot_world_up > 0.95 and np.linalg.norm(local_ang_vel) < 0.5:
                self._state = "COMPLETE"

        return controls

    def is_complete(self, gs: GameState) -> bool:
        """
        The recovery is complete if the car is on the ground and upright,
        or if the state machine explicitly marks it complete.
        """
        return self._state == "COMPLETE"

    def is_invalid(self, gs: GameState) -> bool:
        """
        A recovery is rarely invalid, but could be if stuck for too long.
        For simplicity, we'll assume it's always valid until complete.
        """
        return False
