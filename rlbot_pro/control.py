from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Controls:
    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    boost: bool = False
    jump: bool = False
    handbrake: bool = False

    def clamped(self) -> Controls:
        return Controls(
            throttle=_clamp(self.throttle),
            steer=_clamp(self.steer),
            pitch=_clamp(self.pitch),
            yaw=_clamp(self.yaw),
            roll=_clamp(self.roll),
            boost=bool(self.boost),
            jump=bool(self.jump),
            handbrake=bool(self.handbrake),
        )


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


__all__ = ["Controls"]
