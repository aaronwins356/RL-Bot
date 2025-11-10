# RLGym Migration Guide

## Overview

This repository has been migrated from the legacy RLGym ecosystem to the modern unified `rlgym_rocket_league` package.

## What Changed?

### Old Architecture (Deprecated)
```
rlgym                    # Legacy base package
rlgym-sim               # Simulation package
rlgym-rocket-league     # Rocket League specific
```

### New Architecture (Current)
```
rlgym_rocket_league     # Unified package containing everything
```

## Installation

### Quick Start
```bash
pip install rlgym_rocket_league>=2.0.1
```

### Full Setup
```bash
# Run the automated setup script
python setup_env.py

# Or install manually
pip install -r requirements.txt
```

## Backward Compatibility

The migration includes **full backward compatibility** through compatibility shims:

### Legacy Import Support
```python
# Old code (still works!)
import rlgym
from rlgym_sim.envs import RLGymSimEnv

# Automatically redirects to modern API with deprecation warning
```

### Modern Imports (Recommended)
```python
# New code (recommended)
import rlgym_rocket_league

# Create environment
env = rlgym_rocket_league.make(
    team_size=1,
    tick_skip=8,
    spawn_opponents=True
)
```

## Compatibility Shims

Two compatibility modules provide seamless backward compatibility:

### 1. `rlgym/__init__.py`
- Redirects all `import rlgym` to `rlgym_rocket_league`
- Issues deprecation warning
- Fully transparent to existing code

### 2. `rlgym_sim/__init__.py`
- Redirects all `import rlgym_sim` to `rlgym_rocket_league`
- Provides `RLGymSimEnv` as alias for `rlgym_rocket_league.make`
- Supports all utility submodules

## Import Patterns

All core files now use this pattern:

```python
# Try modern import first
try:
    from rlgym_rocket_league.rocket_league.api import GameState, Car
except ImportError:
    # Fallback to compatibility shim
    from rlgym.rocket_league.api import GameState, Car
```

This ensures:
- ✅ Modern package is used when available
- ✅ Legacy code works without modifications
- ✅ Clear migration path with warnings
- ✅ No breaking changes

## Verification

### Check Environment
```bash
python verify_rl_env.py
```

Expected output:
```
✓ Import PyTorch
✓ Import Gymnasium  
✓ Import NumPy
✓ Import RLGym-Rocket-League (modern)
✓ Import RLGym (legacy compatibility)
✓ Import RLGym-Sim (legacy compatibility)
...
✅ All checks passed (13/13)
```

### Test Training
```bash
# Using main.py (legacy)
python main.py

# Using RocketMind (modern)
python -m rocketmind.main train
```

## Files Modified

### Core Changes
- ✅ `requirements.txt` - Updated package dependencies
- ✅ `verify_rl_env.py` - Tests modern and legacy imports
- ✅ `setup_env.py` - Installs correct packages and creates shims

### Compatibility Shims (New)
- ✅ `rlgym/__init__.py` - Legacy rlgym compatibility
- ✅ `rlgym_sim/__init__.py` - Legacy rlgym_sim compatibility
- ✅ `rlgym_sim/utils/*.py` - Utility module compatibility

### Core Module Updates
- ✅ `rl_bot/core/env_setup.py` - Modern imports with fallback
- ✅ `rl_bot/core/agent.py` - Modern imports with fallback
- ✅ `rl_bot/core/behaviors.py` - Modern imports with fallback
- ✅ `rl_bot/core/advanced_obs.py` - Modern imports with fallback
- ✅ `rl_bot/core/reward_functions.py` - Modern imports with fallback
- ✅ `rocketmind/envs/rocket_env.py` - Modern imports with fallback

## API Changes

### Environment Creation

**Old (still works):**
```python
from rlgym_sim.envs import RLGymSimEnv

env = RLGymSimEnv(
    reward_fn=reward_fn,
    obs_builder=obs_builder
)
```

**New (recommended):**
```python
import rlgym_rocket_league

env = rlgym_rocket_league.make(
    reward_fn=reward_fn,
    obs_builder=obs_builder,
    team_size=1,
    tick_skip=8
)
```

### Imports

**Old (still works):**
```python
from rlgym.api import RLGym
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.action_parsers import LookupTableAction
```

**New (recommended):**
```python
from rlgym_rocket_league.api import RLGym
from rlgym_rocket_league.rocket_league.api import GameState, Car
from rlgym_rocket_league.rocket_league.action_parsers import LookupTableAction
```

## Troubleshooting

### Issue: Import errors
**Solution:** Run `python setup_env.py` to install packages and create shims

### Issue: "Cannot import rlgym_rocket_league"
**Solution:** 
```bash
pip install rlgym_rocket_league>=2.0.1
```

### Issue: Deprecation warnings
**Status:** Expected behavior. These warnings guide you to update imports.

To suppress temporarily:
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Issue: Environment creation fails
**Solution:** Check that `rlgym_rocket_league` is installed correctly:
```bash
python -c "import rlgym_rocket_league; print(rlgym_rocket_league.__version__)"
```

## Migration Checklist

For developers updating code:

- [ ] Install `rlgym_rocket_league>=2.0.1`
- [ ] Run `python verify_rl_env.py` to ensure 13/13 checks pass
- [ ] Test environment creation
- [ ] Test training startup
- [ ] Update imports in new code to use `rlgym_rocket_league` directly
- [ ] Remove suppression of deprecation warnings (if any)
- [ ] Update documentation

## Benefits of Migration

1. **Unified API** - One package instead of three
2. **Better Performance** - Optimized RocketSim integration
3. **Active Development** - Modern package is actively maintained
4. **Cleaner Imports** - Simpler import structure
5. **Future-Proof** - Legacy packages are no longer maintained

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Run `python verify_rl_env.py` for diagnostics
3. Check that all dependencies are installed: `pip list | grep rlgym`
4. Review deprecation warnings for guidance

## References

- **RLGym-Rocket-League**: Modern unified package
- **Compatibility Shims**: Automatic redirection for legacy imports
- **Verification Script**: `verify_rl_env.py` - comprehensive testing
- **Setup Script**: `setup_env.py` - automated installation

---

**Migration Status:** ✅ Complete

**Backward Compatibility:** ✅ Full support through shims

**Ready for Production:** ✅ Yes
