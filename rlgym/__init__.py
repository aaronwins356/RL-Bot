"""
Compatibility shim for legacy 'rlgym' imports.
Redirects to 'rlgym_rocket_league' automatically.

This module provides backward compatibility for code that used the deprecated
rlgym package. All imports are redirected to the new rlgym_rocket_league.
"""

import sys
import importlib
import warnings

# Issue deprecation warning
warnings.warn(
    "The 'rlgym' package is deprecated and has been replaced by 'rlgym_rocket_league'. "
    "Please update your imports to use 'rlgym_rocket_league' instead of 'rlgym'. "
    "This compatibility shim will redirect imports automatically.",
    DeprecationWarning,
    stacklevel=2
)

# Try to import rlgym_rocket_league
try:
    _rlgym_rl = importlib.import_module("rlgym_rocket_league")
    
    # Redirect this module to rlgym_rocket_league
    sys.modules["rlgym"] = _rlgym_rl
    
    # Update globals to include all exports from rlgym_rocket_league
    globals().update(vars(_rlgym_rl))
    
    # Export common names
    if hasattr(_rlgym_rl, 'make'):
        make = _rlgym_rl.make
    
    __all__ = list(_rlgym_rl.__dict__.keys())
    
except ImportError as e:
    raise ImportError(
        "rlgym_rocket_league is not installed. "
        "Install it with: pip install rlgym_rocket_league>=2.0.1"
    ) from e
