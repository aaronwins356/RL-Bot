"""Telemetry data structures for the ProBot GUI integration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


Vector3 = Tuple[float, float, float]


@dataclass
class ControlOutput:
    """Represents the control commands the bot would send to Rocket League."""

    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    boost: bool = False
    jump: bool = False
    handbrake: bool = False


@dataclass
class PhysicsState:
    """Stores core physics state for the car or ball."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class CarState(PhysicsState):
    """In-memory representation of the bot's car physics."""

    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    boost: float = 100.0
    has_jumped: bool = False
    last_jump_time: float = 0.0


@dataclass
class BallState(PhysicsState):
    """Minimal data needed for deterministic ball simulation."""


@dataclass
class BallPredictionSlice:
    """Single time slice of a predicted ball trajectory."""

    time: float
    position: Vector3
    velocity: Vector3


@dataclass
class BotTelemetry:
    """Aggregated telemetry published to the GUI."""

    speed: float
    boost: float
    shot_accuracy: float
    mechanic: str
    position: Vector3
    target: Vector3
    ball_prediction: List[BallPredictionSlice]
    controls: ControlOutput
    extra_metrics: Dict[str, float] = field(default_factory=dict)
