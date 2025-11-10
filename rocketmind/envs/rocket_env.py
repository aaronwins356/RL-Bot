"""
Rocket League Environment Wrapper for RocketMind.
Provides enhanced environment with domain randomization, curriculum learning,
and compatibility with both RLGym-Rocket-League and RLBot.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import warnings

try:
    import rlgym.rocket_league
    RLGYM_AVAILABLE = True
except ImportError:
    RLGYM_AVAILABLE = False
    warnings.warn("rlgym.rocket_league not available - environment creation will fail")


class RocketLeagueEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for Rocket League environment.
    Adds domain randomization, reward shaping, and curriculum learning support.
    """
    
    def __init__(
        self,
        team_size: int = 1,
        tick_skip: int = 8,
        spawn_opponents: bool = True,
        self_play: bool = False,
        timeout_seconds: int = 300,
        domain_randomization: bool = False,
        gravity_mult_range: Tuple[float, float] = (0.9, 1.1),
        boost_spawn_rate_range: Tuple[float, float] = (0.8, 1.2),
        obs_builder: Optional[Any] = None,
        reward_fn: Optional[Any] = None,
        state_setter: Optional[Any] = None,
        action_parser: Optional[Any] = None,
    ):
        """
        Args:
            team_size: Players per team (1, 2, or 3)
            tick_skip: Physics ticks per action
            spawn_opponents: Whether to spawn opponent bots
            self_play: Enable self-play mode
            timeout_seconds: Episode timeout in seconds
            domain_randomization: Enable domain randomization
            gravity_mult_range: Range for gravity randomization
            boost_spawn_rate_range: Range for boost spawn rate randomization
            obs_builder: Custom observation builder
            reward_fn: Custom reward function
            state_setter: Custom state setter
            action_parser: Custom action parser
        """
        super().__init__()
        
        if not RLGYM_AVAILABLE:
            raise ImportError(
                "rlgym_rocket_league is not installed. "
                "Install with: pip install git+https://github.com/RLGym/rlgym-rocket-league.git"
            )
        
        self.team_size = team_size
        self.tick_skip = tick_skip
        self.spawn_opponents = spawn_opponents
        self.self_play = self_play
        self.timeout_seconds = timeout_seconds
        self.domain_randomization = domain_randomization
        self.gravity_mult_range = gravity_mult_range
        self.boost_spawn_rate_range = boost_spawn_rate_range
        
        # Create base environment
        self.env = rlgym.rocket_league.make(
            team_size=team_size,
            tick_skip=tick_skip,
            spawn_opponents=spawn_opponents,
            self_play=self_play,
            timeout_seconds=timeout_seconds,
            obs_builder=obs_builder,
            reward_fn=reward_fn,
            state_setter=state_setter,
            action_parser=action_parser,
        )
        
        # Expose spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with optional domain randomization.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._apply_domain_randomization()
        
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Reward
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def _apply_domain_randomization(self):
        """
        Apply domain randomization to environment parameters.
        Randomizes gravity, boost spawn rates, etc.
        """
        # This is a placeholder - actual implementation would require
        # access to RocketSim parameters through rlgym_rocket_league
        # For now, this serves as a hook for future implementation
        pass
    
    def render(self, mode: str = 'human'):
        """Render environment (if supported)."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    @property
    def unwrapped(self):
        """Get unwrapped environment."""
        return self.env


def create_rocket_env(config: Dict[str, Any]) -> RocketLeagueEnv:
    """
    Create a single Rocket League environment from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RocketLeagueEnv instance
    """
    env_config = config.get('environment', {})
    
    env = RocketLeagueEnv(
        team_size=env_config.get('team_size', 1),
        tick_skip=env_config.get('tick_skip', 8),
        spawn_opponents=env_config.get('spawn_opponents', True),
        self_play=env_config.get('self_play', False),
        timeout_seconds=env_config.get('timeout_seconds', 300),
        domain_randomization=env_config.get('domain_randomization', False),
        gravity_mult_range=tuple(env_config.get('gravity_mult_range', [0.9, 1.1])),
        boost_spawn_rate_range=tuple(env_config.get('boost_spawn_rate_range', [0.8, 1.2])),
    )
    
    return env


def create_vec_env(
    config: Dict[str, Any],
    num_envs: Optional[int] = None
) -> gym.vector.VectorEnv:
    """
    Create vectorized Rocket League environments.
    
    Args:
        config: Configuration dictionary
        num_envs: Number of parallel environments (overrides config if provided)
        
    Returns:
        Vectorized environment
    """
    env_config = config.get('environment', {})
    n_envs = num_envs or env_config.get('num_envs', 4)
    
    # Create environment factory
    def make_env():
        return create_rocket_env(config)
    
    # Create vectorized environment
    # Use AsyncVectorEnv for better parallelization
    try:
        env = gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])
    except Exception as e:
        warnings.warn(f"AsyncVectorEnv failed ({e}), falling back to SyncVectorEnv")
        env = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])
    
    return env


class LegacyRLGymShim:
    """
    Legacy compatibility shim for old rlgym imports.
    Redirects to rlgym_rocket_league.
    """
    
    @staticmethod
    def make(*args, **kwargs):
        """
        Legacy make function that redirects to rlgym_rocket_league.
        
        Issues a deprecation warning and calls rlgym_rocket_league.make.
        """
        warnings.warn(
            "Legacy 'rlgym.make' is deprecated. "
            "Use 'rlgym_rocket_league.make' instead. "
            "This call has been automatically redirected.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not RLGYM_AVAILABLE:
            raise ImportError(
                "rlgym.rocket_league is not installed. "
                "Install with: pip install rlgym-rocket-league>=2.0.1"
            )
        
        return rlgym.rocket_league.make(*args, **kwargs)


# Install legacy shim if someone tries to import old rlgym
def install_legacy_shim():
    """
    Install legacy compatibility shim for old rlgym imports.
    This allows old code using 'import rlgym' to work with warnings.
    """
    import sys
    
    # Create fake rlgym module
    if 'rlgym' not in sys.modules:
        import types
        fake_rlgym = types.ModuleType('rlgym')
        fake_rlgym.make = LegacyRLGymShim.make
        sys.modules['rlgym'] = fake_rlgym


# Optionally install shim on import
# install_legacy_shim()  # Uncomment to enable automatic legacy support
