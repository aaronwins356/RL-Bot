"""
Compatibility shim for rlgym_sim.utils.
Redirects to rlgym_rocket_league.rocket_league.utils.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils is deprecated. Use rlgym_rocket_league.rocket_league.utils instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
try:
    from rlgym_rocket_league.rocket_league.utils import *
    from rlgym_rocket_league.rocket_league import utils as _utils
except ImportError:
    # Fallback to compatibility shim
    from rlgym.rocket_league.utils import *
    from rlgym.rocket_league import utils as _utils

# Make submodules available
gamestates = _utils.gamestates if hasattr(_utils, 'gamestates') else None
state_setters = _utils.state_setters if hasattr(_utils, 'state_setters') else None
obs_builders = _utils.obs_builders if hasattr(_utils, 'obs_builders') else None
action_parsers = _utils.action_parsers if hasattr(_utils, 'action_parsers') else None
reward_functions = _utils.reward_functions if hasattr(_utils, 'reward_functions') else None
terminal_conditions = _utils.terminal_conditions if hasattr(_utils, 'terminal_conditions') else None

__all__ = [
    'gamestates',
    'state_setters', 
    'obs_builders',
    'action_parsers',
    'reward_functions',
    'terminal_conditions'
]
