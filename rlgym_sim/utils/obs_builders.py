"""
Compatibility shim for rlgym_sim.utils.obs_builders.
Redirects to rlgym.rocket_league.utils.obs_builders.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.obs_builders is deprecated. Use rlgym.rocket_league.utils.obs_builders instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils.obs_builders import *

__all__ = ['ObsBuilder']
