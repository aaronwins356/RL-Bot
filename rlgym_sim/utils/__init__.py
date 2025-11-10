"""
Compatibility shim for rlgym_sim.utils.
Redirects to rlgym.rocket_league.utils.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils is deprecated. Use rlgym.rocket_league.utils instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils import *
from rlgym.rocket_league import utils as _utils

# Make submodules available
gamestates = _utils.gamestates
state_setters = _utils.state_setters
obs_builders = _utils.obs_builders
action_parsers = _utils.action_parsers
reward_functions = _utils.reward_functions
terminal_conditions = _utils.terminal_conditions

__all__ = [
    'gamestates',
    'state_setters', 
    'obs_builders',
    'action_parsers',
    'reward_functions',
    'terminal_conditions'
]
