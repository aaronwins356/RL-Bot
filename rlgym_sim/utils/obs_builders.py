"""
Compatibility shim for rlgym_sim.utils.obs_builders.
Redirects to rlgym_rocket_league.utils.obs_builders.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.obs_builders is deprecated. Use rlgym_rocket_league.utils.obs_builders instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
from rlgym_rocket_league.utils.obs_builders import *

__all__ = ['ObsBuilder']
