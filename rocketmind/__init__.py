"""
RocketMind - Next-generation PPO Rocket League Bot
"""

__version__ = "1.0.0"
__author__ = "RocketMind Team"

from . import ppo_core
from . import rlbot_interface
from . import visualization

__all__ = ['ppo_core', 'rlbot_interface', 'visualization']
