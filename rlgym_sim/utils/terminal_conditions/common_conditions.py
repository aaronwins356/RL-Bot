"""
Compatibility shim for rlgym_sim.utils.terminal_conditions.common_conditions.
Redirects to rlgym.rocket_league.utils.terminal_conditions.common_conditions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.terminal_conditions.common_conditions is deprecated. "
    "Use rlgym.rocket_league.utils.terminal_conditions.common_conditions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils.terminal_conditions.common_conditions import *

__all__ = ['TimeoutCondition', 'GoalScoredCondition']
