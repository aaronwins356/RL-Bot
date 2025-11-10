"""
Compatibility shim for legacy rlgym_sim imports.
Redirects to rlgym.rocket_league 2.0+ unified API.

This module provides backward compatibility for code that used the deprecated
rlgym-sim package. All imports are redirected to the new rlgym.rocket_league.
"""

import warnings
import sys

# Issue deprecation warning
warnings.warn(
    "rlgym_sim is deprecated and has been replaced by rlgym.rocket_league 2.0+. "
    "All functionality has been unified into the rlgym.rocket_league package. "
    "Please update your imports to use 'rlgym.rocket_league' instead of 'rlgym_sim'.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export main functionality from rlgym.rocket_league
try:
    from rlgym.rocket_league import make
    from rlgym.rocket_league import utils
    
    # Make these available at package level
    __all__ = ['make', 'utils', 'envs']
    
except ImportError as e:
    raise ImportError(
        "rlgym.rocket_league is not installed. "
        "Install it with: pip install rlgym-rocket-league>=2.0.1"
    ) from e


# Create envs submodule for backward compatibility
class _EnvsModule:
    """Fake envs module that redirects to rlgym.rocket_league."""
    
    @staticmethod
    def RLGymSimEnv(*args, **kwargs):
        """Legacy RLGymSimEnv - redirects to rlgym.rocket_league.make."""
        warnings.warn(
            "RLGymSimEnv is deprecated. Use rlgym.rocket_league.make() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return make(*args, **kwargs)


# Install fake envs module
envs = _EnvsModule()
sys.modules['rlgym_sim.envs'] = envs
