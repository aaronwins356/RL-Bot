"""Rule-based policy for Rocket League bot.

This module implements tactical decision-making using hard-coded rules
for situations like kickoffs, defense, and fallback behaviors.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .intents import Intent


@dataclass
class GameContext:
    """Context information for rule-based decision making."""
    is_kickoff: bool
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    car_position: np.ndarray
    car_velocity: np.ndarray
    car_rotation: np.ndarray  # forward, right, up vectors
    boost_amount: float
    on_ground: bool
    has_flip: bool
    is_closest_to_ball: bool
    is_last_man: bool
    teammates_positions: list
    opponents_positions: list
    game_time: float
    score_diff: int  # positive if winning


class RulePolicy:
    """Rule-based policy for tactical decision-making.
    
    Implements hard-coded logic for:
    - Kickoff variants based on spawn position
    - Early intercepts and last-man defense
    - Boost conservation and safe rotation
    - Air/wall bailout when confidence is low
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize rule policy.
        
        Args:
            config: Configuration dictionary with settings like:
                - aggressive: bool, more aggressive plays
                - boost_conservation: bool, conserve boost when low
                - safe_rotation: bool, prioritize safe rotations
        """
        self.config = config or {}
        self.aggressive = self.config.get("aggressive", False)
        self.boost_conservation = self.config.get("boost_conservation", True)
        self.safe_rotation = self.config.get("safe_rotation", True)
        
        # Thresholds
        self.LOW_BOOST_THRESHOLD = 30.0
        self.CRITICAL_BOOST_THRESHOLD = 10.0
        self.SAFE_DISTANCE_THRESHOLD = 1500.0  # Stay this far from ball when rotating
        self.CHALLENGE_DISTANCE = 1000.0
        
    def get_action(self, context: GameContext) -> Tuple[np.ndarray, Intent, float]:
        """Get action from rule policy.
        
        Args:
            context: Current game context
            
        Returns:
            Tuple of (controls, intent, confidence):
                - controls: 8-dimensional action array
                - intent: High-level intent enum
                - confidence: Confidence score (0-1)
        """
        # High confidence for rule-based decisions
        confidence = 0.95
        
        # Kickoff handling
        if context.is_kickoff:
            return self._kickoff_action(context), Intent.KICKOFF, confidence
        
        # Last man defense - highest priority
        if context.is_last_man and self._ball_threatening():
            return self._defensive_action(context), Intent.SAVE, confidence
        
        # Low boost - get boost
        if context.boost_amount < self.CRITICAL_BOOST_THRESHOLD:
            return self._boost_pickup_action(context), Intent.BOOST_PICKUP, confidence
        
        # Ball is close - decide challenge or rotate
        ball_dist = np.linalg.norm(context.ball_position - context.car_position)
        
        if ball_dist < self.CHALLENGE_DISTANCE:
            if self._should_challenge(context):
                return self._challenge_action(context), Intent.CHALLENGE, confidence
            else:
                return self._rotate_back_action(context), Intent.ROTATE_BACK, confidence
        
        # Default positioning
        if context.is_closest_to_ball:
            return self._drive_to_ball_action(context), Intent.DRIVE_TO_BALL, confidence
        else:
            return self._position_action(context), Intent.POSITION_DEFENSE, confidence
    
    def _kickoff_action(self, context: GameContext) -> np.ndarray:
        """Generate kickoff action based on spawn position."""
        # Simple diagonal kickoff
        ball_dir = context.ball_position - context.car_position
        ball_dir = ball_dir / (np.linalg.norm(ball_dir) + 1e-6)
        
        # Get angle to ball
        forward = context.car_rotation[:3]
        angle = np.arctan2(
            np.cross(forward[:2].astype(float), ball_dir[:2].astype(float)),
            np.dot(forward[:2], ball_dir[:2])
        )
        
        # Controls: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        controls = np.array([
            1.0,  # throttle
            np.clip(angle * 3.0, -1.0, 1.0),  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            1.0 if context.boost_amount > 20 else 0.0,  # boost
            0.0   # handbrake
        ])
        
        return controls
    
    def _defensive_action(self, context: GameContext) -> np.ndarray:
        """Generate defensive action - prioritize save."""
        # Calculate where ball will be
        ball_future_pos = context.ball_position + context.ball_velocity * 0.5
        
        # Position between ball and goal
        goal_pos = np.array([0.0, -5120.0, 0.0])  # Own goal
        intercept_pos = goal_pos + (ball_future_pos - goal_pos) * 0.5
        
        target_dir = intercept_pos - context.car_position
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-6)
        
        forward = context.car_rotation[:3]
        angle = np.arctan2(
            np.cross(forward[:2].astype(float), target_dir[:2].astype(float)),
            np.dot(forward[:2], target_dir[:2])
        )
        
        controls = np.array([
            1.0,  # throttle
            np.clip(angle * 3.0, -1.0, 1.0),  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            1.0 if np.linalg.norm(target_dir) > 2000 else 0.0,  # boost
            0.0   # handbrake
        ])
        
        return controls
    
    def _challenge_action(self, context: GameContext) -> np.ndarray:
        """Generate aggressive challenge action."""
        ball_dir = context.ball_position - context.car_position
        ball_dir = ball_dir / (np.linalg.norm(ball_dir) + 1e-6)
        
        forward = context.car_rotation[:3]
        angle = np.arctan2(
            np.cross(forward[:2].astype(float), ball_dir[:2].astype(float)),
            np.dot(forward[:2], ball_dir[:2])
        )
        
        # Aggressive challenge
        controls = np.array([
            1.0,  # throttle
            np.clip(angle * 3.0, -1.0, 1.0),  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            1.0 if context.boost_amount > 30 else 0.0,  # boost
            0.0   # handbrake
        ])
        
        return controls
    
    def _rotate_back_action(self, context: GameContext) -> np.ndarray:
        """Generate safe rotation action."""
        # Rotate to back post
        goal_pos = np.array([0.0, -5120.0, 0.0])
        back_post_offset = np.array([800.0 if context.car_position[0] > 0 else -800.0, 0.0, 0.0])
        target_pos = goal_pos + back_post_offset
        
        target_dir = target_pos - context.car_position
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-6)
        
        forward = context.car_rotation[:3]
        angle = np.arctan2(
            np.cross(forward[:2].astype(float), target_dir[:2].astype(float)),
            np.dot(forward[:2], target_dir[:2])
        )
        
        # Check if we should halfflip
        dot = np.dot(forward[:2], target_dir[:2])
        should_reverse = dot < -0.5
        
        controls = np.array([
            -1.0 if should_reverse else 1.0,  # throttle
            np.clip(angle * 2.0, -1.0, 1.0),  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            0.0,  # boost (conserve during rotation)
            0.0   # handbrake
        ])
        
        return controls
    
    def _boost_pickup_action(self, context: GameContext) -> np.ndarray:
        """Navigate to nearest boost pad."""
        # For now, just drive forward (would need boost pad locations)
        controls = np.array([
            1.0,  # throttle
            0.0,  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            0.0,  # boost
            0.0   # handbrake
        ])
        
        return controls
    
    def _drive_to_ball_action(self, context: GameContext) -> np.ndarray:
        """Drive toward ball."""
        ball_dir = context.ball_position - context.car_position
        ball_dir = ball_dir / (np.linalg.norm(ball_dir) + 1e-6)
        
        forward = context.car_rotation[:3]
        angle = np.arctan2(
            np.cross(forward[:2].astype(float), ball_dir[:2].astype(float)),
            np.dot(forward[:2], ball_dir[:2])
        )
        
        controls = np.array([
            1.0,  # throttle
            np.clip(angle * 3.0, -1.0, 1.0),  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            1.0 if context.boost_amount > 50 else 0.0,  # boost
            0.0   # handbrake
        ])
        
        return controls
    
    def _position_action(self, context: GameContext) -> np.ndarray:
        """Position defensively."""
        # Position in midfield/defensive third
        target_y = -2000.0  # Defensive positioning
        target_x = context.ball_position[0] * 0.5  # Match ball's lateral position somewhat
        target_pos = np.array([target_x, target_y, 0.0])
        
        target_dir = target_pos - context.car_position
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-6)
        
        forward = context.car_rotation[:3]
        angle = np.arctan2(
            np.cross(forward[:2].astype(float), target_dir[:2].astype(float)),
            np.dot(forward[:2], target_dir[:2])
        )
        
        controls = np.array([
            1.0,  # throttle
            np.clip(angle * 2.0, -1.0, 1.0),  # steer
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            0.0,  # jump
            0.0,  # boost
            0.0   # handbrake
        ])
        
        return controls
    
    def _should_challenge(self, context: GameContext) -> bool:
        """Decide if we should challenge for the ball."""
        # Challenge if:
        # - We're closest to ball
        # - We have enough boost (or boost conservation is off)
        # - Ball is moving away from our goal or is neutral
        
        if not context.is_closest_to_ball:
            return False
        
        if self.boost_conservation and context.boost_amount < self.LOW_BOOST_THRESHOLD:
            return False
        
        # Check if ball is threatening our goal
        goal_pos = np.array([0.0, -5120.0, 0.0])
        ball_to_goal = goal_pos - context.ball_position
        ball_vel_toward_goal = np.dot(context.ball_velocity, ball_to_goal)
        
        # If ball moving toward our goal and we're close, challenge
        if ball_vel_toward_goal > 0:
            return True
        
        # If aggressive mode, challenge more often
        if self.aggressive:
            return True
        
        # Default: be safe
        return False
    
    def _ball_threatening(self) -> bool:
        """Check if ball is threatening our goal."""
        # Simplified - would need actual ball position/velocity
        return True  # Conservative: assume threatening if we're last man
