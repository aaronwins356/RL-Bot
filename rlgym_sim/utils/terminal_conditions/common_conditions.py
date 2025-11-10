"""
Compatibility shim for rlgym_sim.utils.terminal_conditions.common_conditions.
Redirects to rlgym.rocket_league.utils.terminal_conditions.common_conditions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.terminal_conditions.common_conditions is deprecated. "
    "Use rlgym_rocket_league.rocket_league.utils.terminal_conditions.common_conditions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
try:
    from rlgym_rocket_league.rocket_league.utils.terminal_conditions.common_conditions import *
except ImportError:
    from rlgym.rocket_league.utils.terminal_conditions.common_conditions import *

__all__ = ['TimeoutCondition', 'GoalScoredCondition']
