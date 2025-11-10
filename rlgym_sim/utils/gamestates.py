"""
Compatibility shim for rlgym_sim.utils.gamestates.
Redirects to rlgym.rocket_league.utils.gamestates.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.gamestates is deprecated. Use rlgym.rocket_league.utils.gamestates instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils.gamestates import *

__all__ = ['GameState', 'PlayerData']
