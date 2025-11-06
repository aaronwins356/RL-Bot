"""RocketSim-based Rocket League environment for training.

This module provides a gym-compatible environment wrapper for RocketSim,
with support for aerial training, boost management, and reward shaping.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import yaml

from core.features.encoder import ObservationEncoder, RawObservation


class RocketSimEnv:
    """Gym-compatible environment for Rocket League using RocketSim.
    
    This environment supports:
    - Aerial training scenarios
    - Boost management
    - Reward shaping for various behaviors
    - Configurable game modes (1v1, 2v2, 3v3)
    
    Attributes:
        observation_space: Observation space (Box)
        action_space: Action space (Box or MultiDiscrete)
        encoder: Observation encoder
        reward_config: Reward shaping configuration
    """
    
    def __init__(
        self,
        reward_config_path: Optional[Path] = None,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        spawn_opponents: bool = True,
        enable_aerial_training: bool = True,
        random_spawn: bool = True,
    ):
        """Initialize RocketSim environment.
        
        Args:
            reward_config_path: Path to reward configuration YAML
            game_mode: Game mode ("1v1", "2v2", "3v3")
            tick_skip: Number of physics ticks per action
            spawn_opponents: Whether to spawn opponent bots
            enable_aerial_training: Enable aerial training scenarios
            random_spawn: Randomize spawn positions
        """
        self.game_mode = game_mode
        self.tick_skip = tick_skip
        self.spawn_opponents = spawn_opponents
        self.enable_aerial_training = enable_aerial_training
        self.random_spawn = random_spawn
        
        # Load reward configuration
        if reward_config_path and reward_config_path.exists():
            with open(reward_config_path, 'r') as f:
                self.reward_config = yaml.safe_load(f)
        else:
            self.reward_config = self._default_reward_config()
        
        # Initialize encoder
        self.encoder = ObservationEncoder(
            normalize=True,
            include_history=False
        )
        
        # Environment state
        self.episode_length = 0
        self.max_episode_length = 3000  # ~2 minutes at 120Hz
        self.total_reward = 0.0
        self.prev_ball_velocity_toward_goal = 0.0
        self.prev_boost = 100.0
        self.last_touch_time = 0.0
        self.aerial_attempts = 0
        self.aerial_successes = 0
        
        # Stats for reward calculation
        self.stats = {
            'goals_scored': 0,
            'goals_conceded': 0,
            'aerial_touches': 0,
            'demos_given': 0,
            'demos_taken': 0,
            'boost_collected': 0,
            'touches': 0,
        }
        
    def _default_reward_config(self) -> Dict[str, Any]:
        """Get default reward configuration."""
        return {
            'sparse': {
                'goal_scored': 10.0,
                'goal_conceded': -10.0,
                'demo_opponent': 1.0,
                'demoed_self': -1.0,
            },
            'dense': {
                'ball_velocity_toward_goal': 0.01,
                'aerial_touch_bonus': 0.5,
                'boost_pickup': 0.1,
                'boost_waste_penalty': -0.01,
                'touch_bonus': 0.1,
                'goal_proximity': 0.05,
            },
            'penalties': {
                'own_goal_risk': -0.5,
                'double_commit': -0.2,
                'missed_aerial': -0.3,
                'boost_starve': -0.1,
            }
        }
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.episode_length = 0
        self.total_reward = 0.0
        self.prev_ball_velocity_toward_goal = 0.0
        self.prev_boost = 100.0
        self.last_touch_time = 0.0
        self.aerial_attempts = 0
        self.aerial_successes = 0
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
        
        # Initialize game state
        # In a real implementation, this would initialize RocketSim
        # For now, return a dummy observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Action to execute (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.episode_length += 1
        
        # Execute action in simulation (placeholder)
        # In real implementation, would apply action to RocketSim
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, action)
        self.total_reward += reward
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.episode_length >= self.max_episode_length
        
        # Additional info
        info = {
            'episode_length': self.episode_length,
            'total_reward': self.total_reward,
            'stats': self.stats.copy(),
            'aerial_success_rate': self.aerial_successes / max(1, self.aerial_attempts),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Encoded observation vector
        """
        # In real implementation, would get state from RocketSim
        # For now, create a dummy raw observation
        raw_obs = RawObservation(
            car_position=np.array([0.0, 0.0, 20.0]),
            car_velocity=np.array([500.0, 0.0, 0.0]),
            car_angular_velocity=np.array([0.0, 0.0, 0.0]),
            car_rotation_matrix=np.eye(3),
            car_boost=self.prev_boost,
            car_on_ground=True,
            car_has_flip=True,
            car_is_demoed=False,
            ball_position=np.array([0.0, 1000.0, 100.0]),
            ball_velocity=np.array([0.0, 500.0, 0.0]),
            ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
            is_kickoff=False,
            game_time=self.episode_length * 0.008,  # 8ms per tick
            score_self=self.stats['goals_scored'],
            score_opponent=self.stats['goals_conceded'],
            game_phase="NEUTRAL"
        )
        
        return self.encoder.encode(raw_obs)
    
    def _calculate_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward for current step.
        
        Args:
            obs: Current observation
            action: Action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Sparse rewards (implemented in subclass with actual game state)
        # These would be triggered by game events
        
        # Dense rewards (shape learning)
        # Ball velocity toward goal
        ball_vel_reward = self.reward_config['dense'].get('ball_velocity_toward_goal', 0.0)
        if ball_vel_reward != 0.0:
            # Would calculate from actual game state
            pass
        
        # Boost management
        boost_pickup_reward = self.reward_config['dense'].get('boost_pickup', 0.0)
        boost_waste_penalty = self.reward_config['dense'].get('boost_waste_penalty', 0.0)
        
        # Aerial rewards
        aerial_bonus = self.reward_config['dense'].get('aerial_touch_bonus', 0.0)
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if episode is terminated
        """
        # Terminate on goal scored
        if self.stats['goals_scored'] > 0 or self.stats['goals_conceded'] > 0:
            return True
        
        return False
    
    def _detect_aerial_opportunity(self, ball_height: float, ball_distance: float) -> bool:
        """Detect if there's an aerial opportunity.
        
        Args:
            ball_height: Height of ball above ground
            ball_distance: Distance to ball
            
        Returns:
            True if aerial opportunity exists
        """
        # Ball must be airborne and reachable
        min_aerial_height = 200.0  # Minimum height to consider aerial
        max_aerial_distance = 2000.0  # Maximum distance to reach
        
        return ball_height > min_aerial_height and ball_distance < max_aerial_distance
    
    def _calculate_boost_efficiency(self, boost_used: float, action_value: float) -> float:
        """Calculate boost usage efficiency.
        
        Args:
            boost_used: Amount of boost consumed
            action_value: Value of action taken (e.g., goal proximity gain)
            
        Returns:
            Efficiency score (0-1)
        """
        if boost_used <= 0:
            return 1.0
        
        # Efficiency is action value per unit boost
        efficiency = action_value / boost_used
        return np.clip(efficiency, 0.0, 1.0)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment.
        
        Args:
            mode: Render mode ('human', 'rgb_array')
            
        Returns:
            Rendered frame if mode='rgb_array', None otherwise
        """
        # In real implementation, would render game state
        pass
    
    def close(self):
        """Close environment and cleanup resources."""
        # Cleanup RocketSim instance
        pass
    
    @property
    def observation_space(self):
        """Get observation space.
        
        Returns:
            Observation space definition
        """
        # Would return gym.spaces.Box
        return None
    
    @property
    def action_space(self):
        """Get action space.
        
        Returns:
            Action space definition
        """
        # Would return gym.spaces.Box or MultiDiscrete
        return None
