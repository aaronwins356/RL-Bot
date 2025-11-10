"""
Compatibility shim for rlgym_sim.utils.reward_functions.
Redirects to rlgym.rocket_league.utils.reward_functions.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.reward_functions is deprecated. Use rlgym.rocket_league.utils.reward_functions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils.reward_functions import *

__all__ = ['RewardFunction', 'CombinedReward']
