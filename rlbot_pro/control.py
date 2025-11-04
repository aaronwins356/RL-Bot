from dataclasses import dataclass


@dataclass(frozen=True)
class Controls:
    """
    Represents the control inputs for the car.
    All values should be between -1.0 and 1.0, except for boolean flags.
    """

    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    boost: bool = False
    jump: bool = False
    handbrake: bool = False

    def __str__(self) -> str:
        """Custom string representation for easier debugging."""
        return (
            f"Controls(T={self.throttle:.2f}, S={self.steer:.2f}, P={self.pitch:.2f}, "
            f"Y={self.yaw:.2f}, R={self.roll:.2f}, B={self.boost}, J={self.jump}, H={self.handbrake})"
        )
