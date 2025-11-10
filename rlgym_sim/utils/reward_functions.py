"""
Compatibility shim for rlgym_sim.utils.reward_functions.
Redirects to rlgym_rocket_league.utils.reward_functions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.reward_functions is deprecated. Use rlgym_rocket_league.utils.reward_functions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
from rlgym_rocket_league.utils.reward_functions import *

__all__ = ['RewardFunction', 'CombinedReward']
