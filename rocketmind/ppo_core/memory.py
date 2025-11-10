"""
Replay buffer and rollout memory for PPO.
High-throughput async-compatible storage.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class RolloutBuffer:
    """
    Storage for PPO rollout data with GAE computation.
    Optimized for vectorized environments.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        num_envs: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Args:
            buffer_size: Steps per environment before update
            obs_dim: Observation dimension
            num_envs: Number of parallel environments
            device: PyTorch device
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Allocate storage
        self.observations = torch.zeros((buffer_size, num_envs, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, num_envs), dtype=torch.long)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        
        # Computed after rollout
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ):
        """Add transition to buffer."""
        self.observations[self.pos] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[self.pos] = torch.as_tensor(action, dtype=torch.long)
        self.rewards[self.pos] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[self.pos] = torch.as_tensor(done, dtype=torch.float32)
        self.values[self.pos] = torch.as_tensor(value, dtype=torch.float32)
        self.log_probs[self.pos] = torch.as_tensor(log_prob, dtype=torch.float32)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute GAE advantages and returns.
        
        Args:
            last_values: Value estimates for last states
            last_dones: Done flags for last states
        """
        last_values = torch.as_tensor(last_values, dtype=torch.float32)
        last_dones = torch.as_tensor(last_dones, dtype=torch.float32)
        
        # Compute advantages with GAE
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Returns are advantages + values
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get all data for training.
        
        Args:
            batch_size: If provided, yields mini-batches
            
        Returns:
            Dictionary of tensors
        """
        # Flatten batch
        indices = np.arange(self.buffer_size * self.num_envs)
        
        data = {
            'observations': self.observations.reshape(-1, self.obs_dim),
            'actions': self.actions.reshape(-1),
            'old_log_probs': self.log_probs.reshape(-1),
            'advantages': self.advantages.reshape(-1),
            'returns': self.returns.reshape(-1),
            'values': self.values.reshape(-1)
        }
        
        if batch_size is None:
            # Return all data
            return {k: v.to(self.device) for k, v in data.items()}
        else:
            # Mini-batch generator
            np.random.shuffle(indices)
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                yield {k: v[batch_indices].to(self.device) for k, v in data.items()}
    
    def reset(self):
        """Reset buffer."""
        self.pos = 0
        self.full = False


class EpisodeBuffer:
    """Buffer for storing complete episodes (for replay and analysis)."""
    
    def __init__(self, max_episodes: int = 1000):
        """
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
    
    def add_episode(self, episode_data: Dict):
        """
        Add complete episode.
        
        Args:
            episode_data: Dictionary with episode information
                - observations: List of observations
                - actions: List of actions
                - rewards: List of rewards
                - total_reward: Sum of rewards
                - length: Episode length
                - info: Additional metadata
        """
        self.episodes.append(episode_data)
    
    def get_recent_episodes(self, n: int = 10) -> List[Dict]:
        """Get n most recent episodes."""
        return list(self.episodes)[-n:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute statistics over stored episodes."""
        if not self.episodes:
            return {}
        
        total_rewards = [ep['total_reward'] for ep in self.episodes]
        lengths = [ep['length'] for ep in self.episodes]
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_length': np.mean(lengths),
            'num_episodes': len(self.episodes)
        }
    
    def clear(self):
        """Clear all episodes."""
        self.episodes.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay (optional advanced feature).
    Useful for off-policy or mixed on/off-policy training.
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension
            alpha: Priority exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta = beta
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add transition with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((obs, action, reward, next_obs, done))
        else:
            self.buffer[self.position] = (obs, action, reward, next_obs, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with priority."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
            weights,
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)
