"""
Reward Functions - Modular reward components for RocketMind.
Port of existing reward system with additional advanced mechanics.
"""

import numpy as np
from typing import Dict, Any


class RewardFunction:
    """Base class for reward components."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        """Calculate reward component."""
        raise NotImplementedError


class GoalReward(RewardFunction):
    """Reward for scoring goals."""
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        if state.get('goal_scored', False):
            return 10.0 * self.weight
        elif state.get('goal_conceded', False):
            return -10.0 * self.weight
        return 0.0


class BallTouchReward(RewardFunction):
    """Reward for touching the ball."""
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        if state.get('ball_touched', False):
            if state.get('is_aerial', False):
                return 1.0 * self.weight  # Aerial touch bonus
            return 0.5 * self.weight
        return 0.0


class VelocityBallToGoalReward(RewardFunction):
    """Reward ball velocity toward opponent goal."""
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        ball_vel_to_goal = state.get('ball_velocity_to_goal', 0.0)
        return ball_vel_to_goal * self.weight


class BoostPickupReward(RewardFunction):
    """Reward for collecting boost."""
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        boost_gained = state.get('boost', 0) - prev_state.get('boost', 0)
        if boost_gained > 0:
            return 0.1 * self.weight
        return 0.0


class DemoReward(RewardFunction):
    """Reward for demolitions."""
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        if state.get('demo_dealt', False):
            return 2.0 * self.weight
        elif state.get('demo_received', False):
            return -2.0 * self.weight
        return 0.0


class PositioningReward(RewardFunction):
    """Reward for good positioning (simplified)."""
    
    def calculate(self, state: Dict[str, Any], prev_state: Dict[str, Any]) -> float:
        # Distance to ball (want to be reasonably close but not always)
        dist_to_ball = state.get('distance_to_ball', 0.0)
        
        # Normalize distance (0-1, where 1 is good positioning)
        optimal_dist = 1000.0  # Game units
        positioning_score = 1.0 - min(abs(dist_to_ball - optimal_dist) / 2000.0, 1.0)
        
        return positioning_score * self.weight * 0.1


class AdaptiveRewardScalper:
    """
    Adaptive reward sculptor that adjusts weights based on performance.
    Experimental feature for dynamic reward shaping.
    """
    
    def __init__(self, initial_weights: Dict[str, float], evolution_rate: float = 0.001):
        self.weights = initial_weights.copy()
        self.evolution_rate = evolution_rate
        self.performance_history = []
    
    def update_weights(self, performance_metric: float):
        """
        Update reward weights based on performance.
        
        Args:
            performance_metric: Current performance (e.g., win rate, average reward)
        """
        self.performance_history.append(performance_metric)
        
        # Simple adaptive logic: increase weights for underperforming areas
        # This is a simplified version - full implementation would be more sophisticated
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            # If performance is stagnating, adjust exploration
            if len(self.performance_history) > 20:
                prev_performance = np.mean(self.performance_history[-20:-10])
                if abs(recent_performance - prev_performance) < 0.01:
                    # Increase exploration-related rewards
                    for key in ['positioning_weight', 'boost_pickup']:
                        if key in self.weights:
                            self.weights[key] *= (1.0 + self.evolution_rate)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()


def create_reward_function(config: Dict[str, Any]) -> Dict[str, RewardFunction]:
    """
    Create reward function from configuration.
    
    Args:
        config: Configuration dictionary with reward weights
        
    Returns:
        reward_components: Dictionary of reward components
    """
    rewards_config = config.get('rewards', {})
    
    components = {
        'goal': GoalReward(weight=1.0),
        'ball_touch': BallTouchReward(weight=rewards_config.get('touch_ball', 0.5)),
        'velocity_ball_to_goal': VelocityBallToGoalReward(
            weight=rewards_config.get('velocity_ball_to_goal', 0.5)
        ),
        'boost_pickup': BoostPickupReward(weight=rewards_config.get('boost_pickup', 0.1)),
        'demo': DemoReward(weight=1.0),
        'positioning': PositioningReward(weight=rewards_config.get('positioning_weight', 0.1))
    }
    
    return components
