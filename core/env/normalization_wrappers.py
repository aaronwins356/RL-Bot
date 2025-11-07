"""Advanced normalization wrappers for observation and reward normalization.

These wrappers implement running statistics for online normalization during training,
which is crucial for stable and efficient reinforcement learning.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import pickle


class RunningMeanStd:
    """Tracks running mean and standard deviation of a stream of values.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """Initialize running statistics.
        
        Args:
            epsilon: Small constant to avoid division by zero
            shape: Shape of the data (empty tuple for scalars)
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data.
        
        Args:
            x: New data (can be batched or scalar)
        """
        # Handle scalar inputs
        if np.isscalar(x):
            x = np.array([[x]])
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from batch statistics.
        
        Args:
            batch_mean: Mean of batch
            batch_var: Variance of batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def save(self, path: Path) -> None:
        """Save statistics to file.
        
        Args:
            path: Path to save pickle file
        """
        state = {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            'epsilon': self.epsilon
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path) -> None:
        """Load statistics from file.
        
        Args:
            path: Path to pickle file
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']
        self.epsilon = state['epsilon']


class VecNormalize(gym.Wrapper):
    """Vectorized environment wrapper for observation and reward normalization.
    
    Normalizes observations and rewards using running statistics.
    Compatible with both single and vectorized environments.
    """
    
    def __init__(
        self,
        env: gym.Env,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """Initialize normalization wrapper.
        
        Args:
            env: Environment to wrap
            training: Whether in training mode (updates statistics)
            norm_obs: Normalize observations
            norm_reward: Normalize rewards
            clip_obs: Clip normalized observations to [-clip_obs, clip_obs]
            clip_reward: Clip normalized rewards to [-clip_reward, clip_reward]
            gamma: Discount factor for reward normalization
            epsilon: Small constant for numerical stability
        """
        super().__init__(env)
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Get observation shape
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_shape = env.observation_space.shape
        else:
            raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")
        
        # Initialize running statistics
        self.obs_rms = RunningMeanStd(shape=obs_shape, epsilon=epsilon)
        self.ret_rms = RunningMeanStd(shape=(), epsilon=epsilon)
        
        # Track return for reward normalization
        self.returns = None
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation.
        
        Args:
            obs: Raw observation
            
        Returns:
            Normalized observation
        """
        if not self.norm_obs:
            return obs
        
        if self.training:
            self.obs_rms.update(obs)
        
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        
        if self.clip_obs > 0:
            normalized = np.clip(normalized, -self.clip_obs, self.clip_obs)
        
        return normalized
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward.
        
        Args:
            reward: Raw reward
            
        Returns:
            Normalized reward
        """
        if not self.norm_reward:
            return reward
        
        # Update return (discounted sum of rewards)
        if self.returns is None:
            self.returns = reward
        else:
            self.returns = self.returns * self.gamma + reward
        
        if self.training:
            self.ret_rms.update(np.array([self.returns]))
        
        normalized = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        
        if self.clip_reward > 0:
            normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)
        
        return normalized
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and normalize observation.
        
        Returns:
            Normalized observation and info dict
        """
        # Reset return tracker
        self.returns = None
        
        result = self.env.reset(**kwargs)
        
        # Handle both old (obs) and new (obs, info) formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        obs = self._normalize_obs(obs)
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and normalize observation and reward.
        
        Args:
            action: Action to take
            
        Returns:
            Normalized observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Normalize observation
        obs = self._normalize_obs(obs)
        
        # Normalize reward
        reward = self._normalize_reward(reward)
        
        # Reset return tracker on episode end
        if terminated or truncated:
            self.returns = None
        
        return obs, reward, terminated, truncated, info
    
    def set_training(self, training: bool) -> None:
        """Set training mode.
        
        Args:
            training: Whether to update running statistics
        """
        self.training = training
    
    def save_running_stats(self, path: Path) -> None:
        """Save running statistics to file.
        
        Args:
            path: Directory to save statistics
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.obs_rms.save(path / 'obs_rms.pkl')
        self.ret_rms.save(path / 'ret_rms.pkl')
    
    def load_running_stats(self, path: Path) -> None:
        """Load running statistics from file.
        
        Args:
            path: Directory containing statistics
        """
        path = Path(path)
        
        obs_path = path / 'obs_rms.pkl'
        ret_path = path / 'ret_rms.pkl'
        
        if obs_path.exists():
            self.obs_rms.load(obs_path)
        
        if ret_path.exists():
            self.ret_rms.load(ret_path)
