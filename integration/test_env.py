"""Mock Rocket League environment for testing.

This module provides a simplified environment for unit testing.
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional


class MockRocketLeagueEnv:
    """Mock environment simulating Rocket League for testing."""
    
    def __init__(self, obs_size: int = 173):
        """Initialize mock environment.
        
        Args:
            obs_size: Size of observation vector
        """
        self.obs_size = obs_size
        self.timestep = 0
        self.max_timesteps = 300  # 5 minutes at 1 tick/frame
        
        # State
        self.ball_position = np.array([0.0, 0.0, 100.0])
        self.car_position = np.array([0.0, -4000.0, 20.0])
        self.score = [0, 0]
        
    def reset(self) -> np.ndarray:
        """Reset environment.
        
        Returns:
            Initial observation
        """
        self.timestep = 0
        self.ball_position = np.array([0.0, 0.0, 100.0])
        self.car_position = np.array([0.0, -4000.0, 20.0])
        self.score = [0, 0]
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action array (8 dimensions)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.timestep += 1
        
        # Simple physics (move car based on action)
        throttle = action[0]
        self.car_position += np.array([0.0, throttle * 100, 0.0])
        
        # Simple reward (closer to ball = better)
        distance = np.linalg.norm(self.ball_position - self.car_position)
        reward = -distance / 1000.0
        
        # Episode ends after max timesteps or if goal scored
        done = self.timestep >= self.max_timesteps
        
        # Random goal
        if np.random.random() < 0.01:
            self.score[0] += 1
            reward += 10.0
            done = True
        
        info = {
            "score": self.score,
            "distance_to_ball": distance
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation vector
        """
        # Simple observation: normalized positions
        obs = np.zeros(self.obs_size)
        
        # Fill in some basic values
        obs[0:3] = self.ball_position / 4096.0
        obs[10:13] = self.car_position / 4096.0
        
        return obs.astype(np.float32)
