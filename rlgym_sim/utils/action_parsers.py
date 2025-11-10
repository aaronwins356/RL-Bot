"""
Compatibility shim for rlgym_sim.utils.action_parsers.
Redirects to rlgym.rocket_league.utils.action_parsers.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.action_parsers is deprecated. Use rlgym.rocket_league.utils.action_parsers instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
from rlgym.rocket_league.utils.action_parsers import *

__all__ = ['ActionParser']
