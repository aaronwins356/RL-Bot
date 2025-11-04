"""Agent output controls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Controls:
    """Controller inputs for a single frame."""

    throttle: float
    steer: float
    pitch: float
    yaw: float
    roll: float
    boost: bool
    jump: bool
    handbrake: bool


__all__ = ["Controls"]
