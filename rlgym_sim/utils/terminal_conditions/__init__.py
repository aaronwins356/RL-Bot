"""
Compatibility shim for rlgym_sim.utils.terminal_conditions.
Redirects to rlgym_rocket_league.utils.terminal_conditions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.terminal_conditions is deprecated. Use rlgym_rocket_league.utils.terminal_conditions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
from rlgym_rocket_league.utils.terminal_conditions import *
from rlgym_rocket_league.utils import terminal_conditions

# Make common_conditions available
common_conditions = terminal_conditions.common_conditions

__all__ = ['common_conditions']
