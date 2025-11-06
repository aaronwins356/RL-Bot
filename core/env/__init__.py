"""Environment module for Rocket League simulation.

This module provides environment classes and wrappers for training RL agents.
"""

from .rocket_sim_env import RocketSimEnv
from .wrappers import NormalizeObservation, FrameStack, RewardShaping

__all__ = [
    'RocketSimEnv',
    'NormalizeObservation',
    'FrameStack',
    'RewardShaping',
]
