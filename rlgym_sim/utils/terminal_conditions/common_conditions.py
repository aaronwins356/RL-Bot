"""
Compatibility shim for rlgym_sim.utils.terminal_conditions.common_conditions.
Redirects to rlgym_rocket_league.utils.terminal_conditions.common_conditions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.terminal_conditions.common_conditions is deprecated. "
    "Use rlgym_rocket_league.utils.terminal_conditions.common_conditions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
from rlgym_rocket_league.utils.terminal_conditions.common_conditions import *

__all__ = ['TimeoutCondition', 'GoalScoredCondition']
