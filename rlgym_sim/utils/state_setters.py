"""
Compatibility shim for rlgym_sim.utils.state_setters.
Redirects to rlgym.rocket_league.utils.state_setters.
"""

import warnings

warnings.warn(
    "rlgym_sim.utils.state_setters is deprecated. Use rlgym_rocket_league.rocket_league.utils.state_setters instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from rlgym.rocket_league
try:
    from rlgym_rocket_league.rocket_league.utils.state_setters import *
except ImportError:
    from rlgym.rocket_league.utils.state_setters import *

__all__ = ['DefaultState', 'StateSetter']
