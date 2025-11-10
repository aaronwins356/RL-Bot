"""
Compatibility shim for rlgym_sim.utils.obs_builders.
Redirects to rlgym.rocket_league.utils.obs_builders.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.obs_builders is deprecated. Use rlgym_rocket_league.rocket_league.utils.obs_builders instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
try:
    from rlgym_rocket_league.rocket_league.utils.obs_builders import *
except ImportError:
    from rlgym.rocket_league.utils.obs_builders import *

__all__ = ['ObsBuilder']
