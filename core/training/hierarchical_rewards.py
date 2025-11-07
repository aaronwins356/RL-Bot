"""Hierarchical reward shaping for advanced mechanics.

Implements reward components from §5 of the problem statement.
"""

from typing import Dict, Any
import numpy as np


class HierarchicalRewardShaper:
    """Reward shaper for hierarchical RL/IL pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward shaper.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        
        # Hit quality rewards
        self.r_contact = config.get('contact', 1.0)
        self.r_speed_bonus = config.get('speed_to_goal_bonus', 0.25)
        self.r_target_cone_bonus = config.get('target_cone_bonus', 0.5)
        self.r_goal_proximity = config.get('goal_proximity', 0.002)
        
        # Setup fidelity
        self.r_fast_aerial_timing = config.get('fast_aerial_timing', 0.2)
        self.r_flip_reset_contact = config.get('flip_reset_contact', 0.4)
        
        # Style bonuses
        self.r_ceiling_shot = config.get('ceiling_bonus', 0.6)
        self.r_musty = config.get('musty_bonus', 0.5)
        self.r_breezi = config.get('breezi_bonus', 0.5)
        self.r_double_tap = config.get('doubletap_bonus', 0.7)
        self.r_flip_reset_goal = config.get('flipreset_goal_bonus', 1.0)
        
        # Costs
        self.c_boost = config.get('boost_cost', -0.004)
        self.c_whiff = config.get('whiff_cost', -0.2)
        self.c_risky_flashy = config.get('risky_flashy_cost', -0.15)
        self.c_bad_recovery = config.get('bad_recovery_cost', -0.3)
        self.c_residual_angular_vel = config.get('residual_angular_vel_cost', -0.05)
        
        # Track state for delta computations
        self.prev_ball_goal_projection = 0.0
        self.prev_boost = 100
        
    def reset(self):
        """Reset reward shaper state."""
        self.prev_ball_goal_projection = 0.0
        self.prev_boost = 100
        
    def compute_reward(self, obs: Dict[str, Any], action: Dict[str, Any],
                      next_obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Compute hierarchical reward.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            info: Additional info (SP name, success flags, etc.)
            
        Returns:
            Total reward
        """
        reward = 0.0
        
        # Hit quality rewards
        if info.get('ball_contact', False):
            reward += self.r_contact
            
            # Speed bonus
            ball_speed_increase = info.get('ball_speed_increase', 0)
            if ball_speed_increase > 800:
                reward += self.r_speed_bonus
            
            # Target cone bonus
            shot_angle = info.get('shot_angle_deg', 90)
            if shot_angle <= 10:
                reward += self.r_target_cone_bonus
        
        # Goal proximity (delta reward)
        ball_pos = next_obs.get('ball_position', np.zeros(3))
        goal_pos = next_obs.get('opponent_goal_position', np.array([0, 5120, 0]))
        ball_to_goal = goal_pos - ball_pos
        projection = -ball_to_goal[1]  # Y-axis toward goal
        
        delta_projection = projection - self.prev_ball_goal_projection
        reward += self.r_goal_proximity * delta_projection
        self.prev_ball_goal_projection = projection
        
        # Setup fidelity rewards
        sp_name = info.get('sp_name', '')
        
        if sp_name == 'SP_FastAerial':
            timing_correct = info.get('fast_aerial_timing_correct', False)
            if timing_correct:
                reward += self.r_fast_aerial_timing
        
        if sp_name == 'SP_FlipReset':
            four_wheel_contact = info.get('four_wheel_contact', False)
            if four_wheel_contact:
                reward += self.r_flip_reset_contact
        
        # Style bonuses (only when OD flagged Flashy OK)
        flashy_ok = info.get('flashy_ok', False)
        
        if flashy_ok:
            if sp_name == 'SP_CeilingShot':
                success = info.get('sp_success', False)
                if success:
                    reward += self.r_ceiling_shot
            
            if sp_name == 'SP_Musty':
                success = info.get('sp_success', False)
                on_target = info.get('shot_on_target', False)
                ball_speed_increase = info.get('ball_speed_increase', 0)
                if success and on_target and ball_speed_increase > 0:
                    reward += self.r_musty
            
            if sp_name == 'SP_Breezi':
                success = info.get('sp_success', False)
                on_target = info.get('shot_on_target', False)
                ball_speed_increase = info.get('ball_speed_increase', 0)
                if success and on_target and ball_speed_increase > 0:
                    reward += self.r_breezi
            
            if sp_name == 'SP_DoubleTap':
                second_contact = info.get('second_contact', False)
                in_window = info.get('in_window', False)
                if second_contact and in_window:
                    reward += self.r_double_tap
            
            if sp_name == 'SP_FlipReset':
                scored_goal = info.get('scored_goal', False)
                if scored_goal:
                    reward += self.r_flip_reset_goal
        
        # Safety / Costs
        boost_used = self.prev_boost - next_obs.get('boost', 0)
        if boost_used > 0:
            reward += self.c_boost * boost_used
        self.prev_boost = next_obs.get('boost', 0)
        
        # Whiff near own box
        whiff = info.get('whiff', False)
        near_own_box = info.get('near_own_box', False)
        if whiff and near_own_box:
            reward += self.c_whiff
        
        # Risky flashy penalty
        risky_flashy = info.get('risky_flashy_attempt', False)
        if risky_flashy:
            reward += self.c_risky_flashy
        
        # Bad recovery
        recovery_time = info.get('recovery_time', 0.0)
        if recovery_time > 1.0:
            reward += self.c_bad_recovery
        
        # Residual angular velocity at landing
        wheels_on_ground = next_obs.get('wheels_on_ground', 0)
        angular_vel = next_obs.get('angular_velocity', np.zeros(3))
        angular_vel_magnitude = np.linalg.norm(angular_vel)
        
        if wheels_on_ground > 0 and angular_vel_magnitude > 0.5:
            # Penalize per 0.1 rad/s
            penalty = (angular_vel_magnitude / 0.1) * self.c_residual_angular_vel
            reward += penalty
        
        return reward
    
    def compute_sparse_rewards(self, info: Dict[str, Any]) -> float:
        """Compute sparse rewards (goals, etc.).
        
        Args:
            info: Info dict
            
        Returns:
            Sparse reward
        """
        reward = 0.0
        
        if info.get('goal_scored', False):
            reward += 10.0
        
        if info.get('goal_conceded', False):
            reward -= 10.0
        
        return reward
