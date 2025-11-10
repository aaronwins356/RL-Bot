"""
Compatibility shim for rlgym_sim.utils.gamestates.
Redirects to rlgym_rocket_league.utils.gamestates.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.gamestates is deprecated. Use rlgym_rocket_league.utils.gamestates instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
from rlgym_rocket_league.utils.gamestates import *

__all__ = ['GameState', 'PlayerData']
