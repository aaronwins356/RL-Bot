"""
Compatibility shim for rlgym_sim.utils.terminal_conditions.
Redirects to rlgym.rocket_league.utils.terminal_conditions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.terminal_conditions is deprecated. Use rlgym.rocket_league.utils.terminal_conditions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils.terminal_conditions import *
from rlgym.rocket_league import utils as _utils

# Make common_conditions available
common_conditions = _utils.terminal_conditions.common_conditions

__all__ = ['common_conditions']
