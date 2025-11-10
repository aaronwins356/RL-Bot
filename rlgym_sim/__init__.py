"""
Compatibility shim for legacy rlgym_sim imports.
Redirects to rlgym_rocket_league 2.0+ unified API.

This module provides backward compatibility for code that used the deprecated
rlgym-sim package. All imports are redirected to the new rlgym_rocket_league.
"""

import warnings
import sys
import importlib
import types

# Issue deprecation warning
warnings.warn(
    "rlgym_sim is deprecated and has been replaced by rlgym_rocket_league 2.0+. "
    "All functionality has been unified into the rlgym_rocket_league package. "
    "Please update your imports to use 'rlgym_rocket_league' instead of 'rlgym_sim'.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export main functionality from rlgym_rocket_league
try:
    _rlgym_rl = importlib.import_module("rlgym_rocket_league")
    
    # Redirect this module to rlgym_rocket_league
    sys.modules["rlgym_sim"] = _rlgym_rl
    
    # Update globals
    globals().update(vars(_rlgym_rl))
    
    # Make common functions available
    if hasattr(_rlgym_rl, 'make'):
        make = _rlgym_rl.make
    
    # Create envs submodule for backward compatibility
    envs = types.SimpleNamespace()
    if hasattr(_rlgym_rl, 'make'):
        envs.RLGymSimEnv = _rlgym_rl.make
    
    sys.modules['rlgym_sim.envs'] = envs
    
    # Make these available at package level
    __all__ = ['make', 'envs'] + list(_rlgym_rl.__dict__.keys())
    
except ImportError as e:
    raise ImportError(
        "rlgym_rocket_league is not installed. "
        "Install it with: pip install rlgym_rocket_league>=2.0.1"
    ) from e
