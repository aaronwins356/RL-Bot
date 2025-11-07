"""Reward shaping and scaling wrappers for environment.

This module implements:
- RewardScaler: normalize rewards to ≈ [-1, 1]
- Dense sub-rewards (ball touch, shots, goals, etc.)
- Combined weighted reward with normalization
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)


class RewardScaler:
    """Normalize rewards to approximately [-1, 1] range.
    
    Uses running mean and std with exponential moving average.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip_range: float = 10.0,
    ):
        """Initialize reward scaler.
        
        Args:
            gamma: Discount factor for return calculation
            epsilon: Small constant for numerical stability
            clip_range: Maximum absolute value for clipped rewards
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # Running statistics
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
        logger.info(f"RewardScaler initialized (clip_range=±{clip_range})")
    
    def update(self, reward: float):
        """Update running statistics with new reward.
        
        Args:
            reward: New reward value
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)
    
    def normalize(self, reward: float, update: bool = True) -> float:
        """Normalize reward using running statistics.
        
        Args:
            reward: Raw reward value
            update: Whether to update statistics
            
        Returns:
            Normalized reward
        """
        if update:
            self.update(reward)
        
        # Calculate standard deviation
        std = np.sqrt(self.var / max(1, self.count - 1)) if self.count > 1 else 1.0
        std = max(std, self.epsilon)
        
        # Normalize
        normalized = (reward - self.mean) / std
        
        # Clip to range
        clipped = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return clipped
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics.
        
        Returns:
            Dictionary with mean, std, count
        """
        std = np.sqrt(self.var / max(1, self.count - 1)) if self.count > 1 else 1.0
        return {
            "mean": self.mean,
            "std": std,
            "count": self.count,
        }


class DenseRewardShaper:
    """Compute dense sub-rewards for better learning signal.
    
    Includes rewards for:
    - Ball touches
    - Shots on goal
    - Goals scored
    - Ball velocity toward enemy net
    - Penalize idle time
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reward shaper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Reward weights
        self.ball_touch_reward = self.config.get("ball_touch_reward", 0.05)
        self.shot_on_goal_reward = self.config.get("shot_on_goal_reward", 0.10)
        self.goal_reward = self.config.get("goal_reward", 1.0)
        self.own_goal_penalty = self.config.get("own_goal_penalty", -1.0)
        self.ball_velocity_weight = self.config.get("ball_velocity_weight", 0.02)
        self.idle_penalty = self.config.get("idle_penalty", -0.001)
        
        # State tracking
        self.last_ball_touch = False
        self.last_ball_position = None
        self.idle_steps = 0
        self.max_idle_steps = self.config.get("max_idle_steps", 30)
        
        logger.info("DenseRewardShaper initialized with weights:")
        logger.info(f"  - Ball touch: {self.ball_touch_reward}")
        logger.info(f"  - Shot on goal: {self.shot_on_goal_reward}")
        logger.info(f"  - Goal: {self.goal_reward}")
        logger.info(f"  - Own goal: {self.own_goal_penalty}")
    
    def compute_reward(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute dense reward based on game state.
        
        Args:
            game_state: Current game state
            info: Additional info dict from environment
            
        Returns:
            Dense reward value
        """
        total_reward = 0.0
        
        # Check for goal scored
        if info.get("goal_scored", False):
            total_reward += self.goal_reward
            logger.debug("Goal scored: +1.0")
        
        # Check for own goal
        if info.get("own_goal", False):
            total_reward += self.own_goal_penalty
            logger.debug("Own goal: -1.0")
        
        # Check for ball touch
        ball_touched = info.get("ball_touched", False)
        if ball_touched and not self.last_ball_touch:
            total_reward += self.ball_touch_reward
            logger.debug(f"Ball touch: +{self.ball_touch_reward}")
            self.idle_steps = 0  # Reset idle counter
        self.last_ball_touch = ball_touched
        
        # Check for shot on goal
        if info.get("shot_on_goal", False):
            total_reward += self.shot_on_goal_reward
            logger.debug(f"Shot on goal: +{self.shot_on_goal_reward}")
        
        # Ball velocity toward enemy goal
        ball_velocity = game_state.get("ball_velocity", np.zeros(3))
        ball_position = game_state.get("ball_position", np.zeros(3))
        
        if self.last_ball_position is not None:
            # Calculate if ball moving toward enemy goal (positive x direction)
            ball_direction = ball_position[0] - self.last_ball_position[0]
            velocity_reward = self.ball_velocity_weight * ball_direction
            total_reward += velocity_reward
        
        # Always ensure ball_position is a numpy array for consistent copying
        if not isinstance(ball_position, np.ndarray):
            ball_position = np.asarray(ball_position)
        self.last_ball_position = ball_position.copy()
        
        # Idle penalty
        if not ball_touched:
            self.idle_steps += 1
            if self.idle_steps > self.max_idle_steps:
                total_reward += self.idle_penalty
        else:
            self.idle_steps = 0
        
        return total_reward
    
    def reset(self):
        """Reset state tracking."""
        self.last_ball_touch = False
        self.last_ball_position = None
        self.idle_steps = 0


class RewardWrapper:
    """Wrapper that combines reward shaping and scaling.
    
    Applies dense sub-rewards and normalizes final reward to std ≈ 1.0.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        normalize: bool = True,
        use_dense_rewards: bool = True,
    ):
        """Initialize reward wrapper.
        
        Args:
            config: Configuration dictionary
            normalize: Whether to normalize rewards
            use_dense_rewards: Whether to use dense reward shaping
        """
        self.config = config or {}
        self.normalize = normalize
        self.use_dense_rewards = use_dense_rewards
        
        # Components
        self.scaler = RewardScaler() if normalize else None
        self.shaper = DenseRewardShaper(config) if use_dense_rewards else None
        
        # Statistics
        self.total_raw_reward = 0.0
        self.total_shaped_reward = 0.0
        self.total_normalized_reward = 0.0
        self.episode_count = 0
        
        logger.info("RewardWrapper initialized:")
        logger.info(f"  - Normalization: {normalize}")
        logger.info(f"  - Dense rewards: {use_dense_rewards}")
    
    def process_reward(
        self,
        raw_reward: float,
        game_state: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Process reward through shaping and normalization.
        
        Args:
            raw_reward: Raw reward from environment
            game_state: Current game state (for dense rewards)
            info: Info dict from environment
            
        Returns:
            Processed reward
        """
        reward = raw_reward
        self.total_raw_reward += raw_reward
        
        # Apply dense reward shaping
        if self.use_dense_rewards and self.shaper:
            if game_state is not None and info is not None:
                dense_reward = self.shaper.compute_reward(game_state, info)
                reward += dense_reward
                self.total_shaped_reward += dense_reward
        
        # Apply normalization
        if self.normalize and self.scaler:
            reward = self.scaler.normalize(reward)
            self.total_normalized_reward += reward
        
        return reward
    
    def reset(self):
        """Reset for new episode."""
        if self.shaper:
            self.shaper.reset()
        self.episode_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward statistics.
        
        Returns:
            Dictionary with reward stats
        """
        stats = {
            "total_raw_reward": self.total_raw_reward,
            "episode_count": self.episode_count,
        }
        
        if self.episode_count > 0:
            stats["avg_raw_reward_per_episode"] = self.total_raw_reward / self.episode_count
        
        if self.use_dense_rewards:
            stats["total_shaped_reward"] = self.total_shaped_reward
            if self.episode_count > 0:
                stats["avg_shaped_reward_per_episode"] = self.total_shaped_reward / self.episode_count
        
        if self.normalize and self.scaler:
            stats.update(self.scaler.get_stats())
            stats["total_normalized_reward"] = self.total_normalized_reward
        
        return stats
