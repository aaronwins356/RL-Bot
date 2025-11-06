"""Environment wrappers for observation and reward processing.

This module provides various wrappers to enhance environment functionality.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from collections import deque


class NormalizeObservation:
    """Wrapper to normalize observations to [-1, 1] range.
    
    Uses running statistics to normalize observations.
    """
    
    def __init__(self, env, epsilon: float = 1e-8):
        """Initialize normalization wrapper.
        
        Args:
            env: Base environment
            epsilon: Small value to avoid division by zero
        """
        self.env = env
        self.epsilon = epsilon
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 0
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return normalized observation."""
        obs = self.env.reset(**kwargs)
        return self._normalize(obs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and normalize observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics.
        
        Args:
            obs: Raw observation
            
        Returns:
            Normalized observation
        """
        # Update running statistics
        if self.obs_mean is None:
            self.obs_mean = obs.copy()
            self.obs_var = np.ones_like(obs)
        else:
            # Incremental mean and variance update
            self.obs_count += 1
            delta = obs - self.obs_mean
            self.obs_mean += delta / self.obs_count
            self.obs_var += delta * (obs - self.obs_mean)
        
        # Normalize
        std = np.sqrt(self.obs_var / max(1, self.obs_count - 1) + self.epsilon)
        normalized = (obs - self.obs_mean) / std
        
        return np.clip(normalized, -10.0, 10.0)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class FrameStack:
    """Wrapper to stack multiple observations for temporal context.
    
    Useful for learning temporal patterns and velocity estimation.
    """
    
    def __init__(self, env, num_stack: int = 4):
        """Initialize frame stacking wrapper.
        
        Args:
            env: Base environment
            num_stack: Number of frames to stack
        """
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and initialize frame stack."""
        obs = self.env.reset(**kwargs)
        
        # Fill stack with initial observation
        for _ in range(self.num_stack):
            self.frames.append(obs)
        
        return self._get_stacked_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.frames.append(obs)
        
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Get stacked observation.
        
        Returns:
            Concatenated observation from all frames
        """
        return np.concatenate(list(self.frames), axis=0)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class RewardShaping:
    """Wrapper to apply custom reward shaping functions.
    
    Allows flexible reward engineering without modifying base environment.
    """
    
    def __init__(
        self,
        env,
        reward_fn: Optional[Callable[[float, Dict[str, Any]], float]] = None,
        reward_scale: float = 1.0,
        reward_clip: Optional[Tuple[float, float]] = None,
    ):
        """Initialize reward shaping wrapper.
        
        Args:
            env: Base environment
            reward_fn: Custom reward function (reward, info) -> shaped_reward
            reward_scale: Scaling factor for rewards
            reward_clip: Optional (min, max) to clip rewards
        """
        self.env = env
        self.reward_fn = reward_fn
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and apply reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply custom reward function if provided
        if self.reward_fn is not None:
            reward = self.reward_fn(reward, info)
        
        # Scale reward
        reward *= self.reward_scale
        
        # Clip reward if specified
        if self.reward_clip is not None:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class AerialTrainingWrapper:
    """Wrapper specifically for aerial training scenarios.
    
    Spawns ball in aerial positions and rewards aerial touches.
    """
    
    def __init__(
        self,
        env,
        aerial_spawn_probability: float = 0.3,
        min_ball_height: float = 300.0,
        max_ball_height: float = 1500.0,
    ):
        """Initialize aerial training wrapper.
        
        Args:
            env: Base environment
            aerial_spawn_probability: Probability of spawning aerial scenario
            min_ball_height: Minimum ball spawn height
            max_ball_height: Maximum ball spawn height
        """
        self.env = env
        self.aerial_spawn_probability = aerial_spawn_probability
        self.min_ball_height = min_ball_height
        self.max_ball_height = max_ball_height
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment with possible aerial spawn."""
        obs = self.env.reset(**kwargs)
        
        # Randomly spawn aerial scenario
        if np.random.random() < self.aerial_spawn_probability:
            self._spawn_aerial_scenario()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment (no modification)."""
        return self.env.step(action)
    
    def _spawn_aerial_scenario(self):
        """Spawn ball in aerial position.
        
        In real implementation, would modify RocketSim state.
        """
        # Would set ball position and velocity for aerial training
        pass
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class BoostManagementWrapper:
    """Wrapper to encourage boost efficiency.
    
    Adds rewards/penalties based on boost usage patterns.
    """
    
    def __init__(
        self,
        env,
        boost_efficiency_weight: float = 0.1,
        boost_starve_penalty: float = -0.2,
        boost_starve_threshold: float = 20.0,
    ):
        """Initialize boost management wrapper.
        
        Args:
            env: Base environment
            boost_efficiency_weight: Weight for boost efficiency reward
            boost_starve_penalty: Penalty when boost is critically low
            boost_starve_threshold: Boost level considered starved
        """
        self.env = env
        self.boost_efficiency_weight = boost_efficiency_weight
        self.boost_starve_penalty = boost_starve_penalty
        self.boost_starve_threshold = boost_starve_threshold
        self.prev_boost = 100.0
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and boost tracking."""
        self.prev_boost = 100.0
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and add boost management rewards."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current boost from info or observation
        current_boost = info.get('boost', 50.0)  # Default if not available
        
        # Calculate boost usage
        boost_used = max(0, self.prev_boost - current_boost)
        
        # Add boost starve penalty
        if current_boost < self.boost_starve_threshold:
            reward += self.boost_starve_penalty
        
        # Add boost efficiency reward if boost was used effectively
        # (would need to measure effectiveness from game state)
        
        self.prev_boost = current_boost
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
