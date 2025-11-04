from typing import Protocol, Tuple

import numpy as np

from rlbot_pro.control import Controls
from rlbot_pro.state import GameState, Vector


class Mechanic(Protocol):
    """
    Protocol for all mechanics.
    Each mechanic must implement these methods.
    """

    def prep(self, gs: GameState, **kwargs) -> None:
        """
        Optional preparation step for the mechanic.
        Can be used to initialize state or calculate targets.
        """
        pass

    def step(self, gs: GameState, **kwargs) -> Controls:
        """
        Calculates the controls needed to execute the mechanic for the current game state.
        """
        ...

    def is_complete(self, gs: GameState) -> bool:
        """
        Checks if the mechanic has completed its objective.
        """
        ...

    def is_invalid(self, gs: GameState) -> bool:
        """
        Checks if the mechanic has become invalid (e.g., ball moved too far, car crashed).
        """
        ...


def get_pitch_yaw_roll(
    car_forward: np.ndarray, car_up: np.ndarray, target_forward: np.ndarray, target_up: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculates the pitch, yaw, and roll needed to orient the car.
    """
    # Calculate yaw
    yaw_angle = np.arctan2(
        target_forward[1] * car_forward[0] - target_forward[0] * car_forward[1],
        target_forward[0] * car_forward[0] + target_forward[1] * car_forward[1],
    )

    # Calculate pitch
    pitch_angle = np.arctan2(target_forward[2], np.linalg.norm(target_forward[:2]))

    # Calculate roll (simplified, might need more robust solution for complex rolls)
    # Project car_up onto the plane perpendicular to target_forward
    projected_car_up = car_up - np.dot(car_up, target_forward) * target_forward
    projected_target_up = target_up - np.dot(target_up, target_forward) * target_forward

    # Angle between projected up vectors
    roll_angle = np.arctan2(
        np.dot(np.cross(projected_car_up, projected_target_up), target_forward),
        np.dot(projected_car_up, projected_target_up),
    )

    return pitch_angle, yaw_angle, roll_angle
