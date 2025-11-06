"""Reward shaping for RLBot training.

This module provides custom dense rewards for various skills:
- Boost control and management
- Demolitions
- Passing and team play
- Goal proximity and scoring
- Positioning and rotation
"""
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class RewardShaper:
    """Shapes rewards to encourage specific behaviors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reward shaper.
        
        Args:
            config: Reward shaping configuration
        """
        self.config = config or {}
        
        # Reward weights
        self.weights = {
            'goal_scored': self.config.get('goal_scored_weight', 10.0),
            'goal_conceded': self.config.get('goal_conceded_weight', -10.0),
            'touch_ball': self.config.get('touch_ball_weight', 0.1),
            'boost_pickup': self.config.get('boost_pickup_weight', 0.05),
            'boost_usage': self.config.get('boost_usage_weight', -0.01),
            'demolition': self.config.get('demolition_weight', 2.0),
            'demolished': self.config.get('demolished_weight', -1.0),
            'goal_proximity': self.config.get('goal_proximity_weight', 0.5),
            'ball_velocity_toward_goal': self.config.get('ball_velocity_weight', 0.3),
            'position_reward': self.config.get('position_reward_weight', 0.2),
            'rotation_reward': self.config.get('rotation_reward_weight', 0.1),
            'aerial_reward': self.config.get('aerial_reward_weight', 0.5),
            'shot_on_goal': self.config.get('shot_on_goal_weight', 1.0),
            'save': self.config.get('save_weight', 2.0),
            'pass_reward': self.config.get('pass_reward_weight', 0.5),
        }
        
        # Track previous state for delta calculations
        self.prev_state = {}
        
        logger.info(f"RewardShaper initialized with {len(self.weights)} reward components")
    
    def compute_reward(
        self,
        game_state: Dict[str, Any],
        agent_id: int,
        prev_game_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute shaped reward for current state.
        
        Args:
            game_state: Current game state
            agent_id: Agent ID
            prev_game_state: Previous game state (for deltas)
            
        Returns:
            Shaped reward value
        """
        reward = 0.0
        reward_breakdown = {}
        
        # Goal scored/conceded
        if 'goals_scored' in game_state:
            if prev_game_state and 'goals_scored' in prev_game_state:
                goals_delta = game_state['goals_scored'] - prev_game_state['goals_scored']
                if goals_delta > 0:
                    reward += self.weights['goal_scored'] * goals_delta
                    reward_breakdown['goal_scored'] = self.weights['goal_scored'] * goals_delta
                
                goals_conceded_delta = game_state.get('goals_conceded', 0) - prev_game_state.get('goals_conceded', 0)
                if goals_conceded_delta > 0:
                    reward += self.weights['goal_conceded'] * goals_conceded_delta
                    reward_breakdown['goal_conceded'] = self.weights['goal_conceded'] * goals_conceded_delta
        
        # Ball touch
        if game_state.get('touched_ball', False):
            reward += self.weights['touch_ball']
            reward_breakdown['touch_ball'] = self.weights['touch_ball']
        
        # Boost management
        boost_reward = self._compute_boost_reward(game_state, prev_game_state)
        if boost_reward != 0:
            reward += boost_reward
            reward_breakdown['boost'] = boost_reward
        
        # Demolitions
        demo_reward = self._compute_demo_reward(game_state, prev_game_state)
        if demo_reward != 0:
            reward += demo_reward
            reward_breakdown['demo'] = demo_reward
        
        # Proximity to ball/goal
        proximity_reward = self._compute_proximity_reward(game_state)
        if proximity_reward != 0:
            reward += proximity_reward
            reward_breakdown['proximity'] = proximity_reward
        
        # Ball velocity toward goal
        velocity_reward = self._compute_ball_velocity_reward(game_state)
        if velocity_reward != 0:
            reward += velocity_reward
            reward_breakdown['ball_velocity'] = velocity_reward
        
        # Positioning reward
        position_reward = self._compute_position_reward(game_state, agent_id)
        if position_reward != 0:
            reward += position_reward
            reward_breakdown['position'] = position_reward
        
        # Aerial reward
        aerial_reward = self._compute_aerial_reward(game_state)
        if aerial_reward != 0:
            reward += aerial_reward
            reward_breakdown['aerial'] = aerial_reward
        
        # Shot on goal
        shot_reward = self._compute_shot_reward(game_state, prev_game_state)
        if shot_reward != 0:
            reward += shot_reward
            reward_breakdown['shot'] = shot_reward
        
        # Save
        save_reward = self._compute_save_reward(game_state, prev_game_state)
        if save_reward != 0:
            reward += save_reward
            reward_breakdown['save'] = save_reward
        
        # Pass
        pass_reward = self._compute_pass_reward(game_state, prev_game_state)
        if pass_reward != 0:
            reward += pass_reward
            reward_breakdown['pass'] = pass_reward
        
        return reward
    
    def _compute_boost_reward(
        self,
        game_state: Dict[str, Any],
        prev_game_state: Optional[Dict[str, Any]]
    ) -> float:
        """Compute boost-related rewards."""
        reward = 0.0
        
        # Reward for picking up boost
        if prev_game_state:
            boost_delta = game_state.get('boost_amount', 0) - prev_game_state.get('boost_amount', 0)
            if boost_delta > 0:
                reward += self.weights['boost_pickup'] * boost_delta / 100.0
            elif boost_delta < 0:
                # Small penalty for boost usage
                reward += self.weights['boost_usage'] * abs(boost_delta) / 100.0
        
        return reward
    
    def _compute_demo_reward(
        self,
        game_state: Dict[str, Any],
        prev_game_state: Optional[Dict[str, Any]]
    ) -> float:
        """Compute demolition-related rewards."""
        reward = 0.0
        
        if prev_game_state:
            demos_delta = game_state.get('demos', 0) - prev_game_state.get('demos', 0)
            if demos_delta > 0:
                reward += self.weights['demolition'] * demos_delta
            
            demolished_delta = game_state.get('demolished', 0) - prev_game_state.get('demolished', 0)
            if demolished_delta > 0:
                reward += self.weights['demolished'] * demolished_delta
        
        return reward
    
    def _compute_proximity_reward(self, game_state: Dict[str, Any]) -> float:
        """Compute proximity-based rewards."""
        reward = 0.0
        
        # Distance to ball
        ball_dist = game_state.get('ball_distance', float('inf'))
        if ball_dist < float('inf'):
            # Reward for being closer to ball (normalized)
            normalized_dist = np.clip(ball_dist / 5000.0, 0, 1)
            reward += self.weights['goal_proximity'] * (1 - normalized_dist) * 0.1
        
        return reward
    
    def _compute_ball_velocity_reward(self, game_state: Dict[str, Any]) -> float:
        """Compute reward for ball velocity toward opponent goal."""
        reward = 0.0
        
        ball_vel_toward_goal = game_state.get('ball_velocity_toward_goal', 0)
        if ball_vel_toward_goal > 0:
            # Reward for hitting ball toward opponent goal
            normalized_vel = np.clip(ball_vel_toward_goal / 3000.0, 0, 1)
            reward += self.weights['ball_velocity_toward_goal'] * normalized_vel
        
        return reward
    
    def _compute_position_reward(self, game_state: Dict[str, Any], agent_id: int) -> float:
        """Compute reward for good positioning."""
        reward = 0.0
        
        # Reward for being in defensive position when needed
        # Reward for being in offensive position when appropriate
        # This would require game state analysis
        
        return reward
    
    def _compute_aerial_reward(self, game_state: Dict[str, Any]) -> float:
        """Compute reward for aerial play."""
        reward = 0.0
        
        # Reward for aerial touches
        if game_state.get('is_aerial', False) and game_state.get('touched_ball', False):
            reward += self.weights['aerial_reward']
        
        return reward
    
    def _compute_shot_reward(
        self,
        game_state: Dict[str, Any],
        prev_game_state: Optional[Dict[str, Any]]
    ) -> float:
        """Compute reward for shots on goal."""
        reward = 0.0
        
        if prev_game_state:
            shots_delta = game_state.get('shots_on_goal', 0) - prev_game_state.get('shots_on_goal', 0)
            if shots_delta > 0:
                reward += self.weights['shot_on_goal'] * shots_delta
        
        return reward
    
    def _compute_save_reward(
        self,
        game_state: Dict[str, Any],
        prev_game_state: Optional[Dict[str, Any]]
    ) -> float:
        """Compute reward for saves."""
        reward = 0.0
        
        if prev_game_state:
            saves_delta = game_state.get('saves', 0) - prev_game_state.get('saves', 0)
            if saves_delta > 0:
                reward += self.weights['save'] * saves_delta
        
        return reward
    
    def _compute_pass_reward(
        self,
        game_state: Dict[str, Any],
        prev_game_state: Optional[Dict[str, Any]]
    ) -> float:
        """Compute reward for passing."""
        reward = 0.0
        
        if prev_game_state:
            passes_delta = game_state.get('passes', 0) - prev_game_state.get('passes', 0)
            if passes_delta > 0:
                reward += self.weights['pass_reward'] * passes_delta
        
        return reward
    
    def get_config(self) -> Dict[str, Any]:
        """Get reward shaping configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'weights': self.weights
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update reward weights.
        
        Args:
            new_weights: New weight values
        """
        self.weights.update(new_weights)
        logger.info(f"Updated reward weights: {new_weights}")


class CurriculumRewardShaper(RewardShaper):
    """Reward shaper that adapts to curriculum stage."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize curriculum reward shaper.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.current_stage = 0
        self.stage_configs = self._create_stage_configs()
    
    def _create_stage_configs(self) -> List[Dict[str, float]]:
        """Create reward configurations for each curriculum stage.
        
        Returns:
            List of weight dictionaries per stage
        """
        return [
            # Stage 0: Basic ground play
            {
                'touch_ball': 0.2,
                'boost_pickup': 0.1,
                'goal_proximity': 0.5,
            },
            # Stage 1: Intermediate mechanics
            {
                'touch_ball': 0.15,
                'boost_pickup': 0.05,
                'boost_usage': -0.02,
                'goal_proximity': 0.3,
                'ball_velocity_toward_goal': 0.5,
            },
            # Stage 2: Advanced positioning
            {
                'position_reward': 0.3,
                'rotation_reward': 0.2,
                'ball_velocity_toward_goal': 0.4,
            },
            # Stage 3: Aerial play
            {
                'aerial_reward': 1.0,
                'shot_on_goal': 1.5,
                'ball_velocity_toward_goal': 0.5,
            },
            # Stage 4: Team play
            {
                'pass_reward': 1.0,
                'position_reward': 0.4,
                'rotation_reward': 0.3,
                'save': 2.5,
            },
        ]
    
    def set_curriculum_stage(self, stage: int):
        """Set current curriculum stage and update weights.
        
        Args:
            stage: Curriculum stage index
        """
        if stage < 0 or stage >= len(self.stage_configs):
            logger.warning(f"Invalid curriculum stage {stage}, ignoring")
            return
        
        self.current_stage = stage
        
        # Update weights based on stage
        stage_weights = self.stage_configs[stage]
        self.update_weights(stage_weights)
        
        logger.info(f"Updated reward weights for curriculum stage {stage}")
