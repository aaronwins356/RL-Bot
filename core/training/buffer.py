"""Replay buffer for storing and sampling experiences.

This module implements an experience replay buffer for RL training.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque


class ReplayBuffer:
    """Experience replay buffer for RL training.
    
    Stores (observation, action, reward, next_observation, done) tuples.
    """
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Statistics
        self.episodes_stored = 0
        self.total_reward = 0.0
    
    def add(
        self,
        experience: Dict[str, Any] = None,
        observation: np.ndarray = None,
        action: np.ndarray = None,
        reward: float = None,
        next_observation: np.ndarray = None,
        done: bool = None,
        info: Optional[Dict[str, Any]] = None
    ):
        """Add experience to buffer.
        
        Args:
            experience: Experience dict (if provided, other args ignored)
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            info: Additional info (optional)
        """
        if experience is not None:
            # Direct dict interface
            self.buffer.append(experience)
            self.total_reward += experience.get('reward', 0.0)
            if experience.get('done', False):
                self.episodes_stored += 1
        else:
            # Legacy interface
            exp = {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
                "info": info or {}
            }
            
            self.buffer.append(exp)
            self.total_reward += reward
            
            if done:
                self.episodes_stored += 1
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary with batched experiences
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": []
        }
        
        for idx in indices:
            exp = self.buffer[idx]
            batch["observations"].append(exp["observation"])
            batch["actions"].append(exp["action"])
            batch["rewards"].append(exp["reward"])
            batch["next_observations"].append(exp["next_observation"])
            batch["dones"].append(exp["done"])
        
        # Convert to numpy arrays
        return {
            "observations": np.array(batch["observations"]),
            "actions": np.array(batch["actions"]),
            "rewards": np.array(batch["rewards"]),
            "next_observations": np.array(batch["next_observations"]),
            "dones": np.array(batch["dones"])
        }
    
    def get_recent_trajectory(self, max_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get most recent trajectory (for on-policy algorithms like PPO).
        
        Args:
            max_length: Maximum trajectory length (None = all recent until done)
            
        Returns:
            Dictionary with trajectory data
        """
        if not self.buffer:
            return {
                "observations": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "next_observations": np.array([]),
                "dones": np.array([])
            }
        
        # Get recent experiences until done or max_length
        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": []
        }
        
        # Iterate backwards to get most recent trajectory
        for i in range(len(self.buffer) - 1, -1, -1):
            exp = self.buffer[i]
            
            trajectory["observations"].insert(0, exp["observation"])
            trajectory["actions"].insert(0, exp["action"])
            trajectory["rewards"].insert(0, exp["reward"])
            trajectory["next_observations"].insert(0, exp["next_observation"])
            trajectory["dones"].insert(0, exp["done"])
            
            if exp["done"]:
                break
            
            if max_length and len(trajectory["observations"]) >= max_length:
                break
        
        # Convert to numpy arrays
        return {
            "observations": np.array(trajectory["observations"]),
            "actions": np.array(trajectory["actions"]),
            "rewards": np.array(trajectory["rewards"]),
            "next_observations": np.array(trajectory["next_observations"]),
            "dones": np.array(trajectory["dones"])
        }
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.episodes_stored = 0
        self.total_reward = 0.0
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer stats
        """
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "episodes_stored": self.episodes_stored,
            "total_reward": self.total_reward,
            "avg_reward_per_episode": (
                self.total_reward / self.episodes_stored
                if self.episodes_stored > 0 else 0.0
            )
        }
