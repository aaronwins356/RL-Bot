"""
Compatibility shim for rlgym_sim.utils.action_parsers.
Redirects to rlgym_rocket_league.rocket_league.utils.action_parsers.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.action_parsers is deprecated. Use rlgym_rocket_league.rocket_league.utils.action_parsers instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym_rocket_league
try:
    from rlgym_rocket_league.rocket_league.utils.action_parsers import *
except ImportError:
    # Fallback to compatibility shim
    from rlgym.rocket_league.utils.action_parsers import *

__all__ = ['ActionParser']
