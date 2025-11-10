"""
Modular behavior system for Rocket League bot.
Provides hardcoded logic for specific situations (kickoffs, recoveries, etc.)
that can override or guide the learned policy.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try modern import first, then fallback to compatibility shim
try:
    from rlgym_rocket_league.rocket_league.api import GameState, Car
except ImportError:
    from rlgym.rocket_league.api import GameState, Car

# For compatibility with old code, alias Car as PlayerData  
PlayerData = Car


@dataclass
class BehaviorAction:
    """
    Represents an action returned by a behavior module.
    If action_index is None, the behavior doesn't want to override.
    """
    action_index: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BehaviorModule:
    """
    Base class for behavior modules that can override policy decisions.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def should_activate(self, player: PlayerData, state: GameState) -> bool:
        """
        Determine if this behavior should activate.
        
        Args:
            player: Current player data
            state: Current game state
            
        Returns:
            True if this behavior should override the policy
        """
        raise NotImplementedError
    
    def get_action(self, player: PlayerData, state: GameState, action_parser) -> BehaviorAction:
        """
        Get the action to take when this behavior is active.
        
        Args:
            player: Current player data
            state: Current game state
            action_parser: Action parser to convert high-level actions
            
        Returns:
            BehaviorAction with action index and metadata
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset behavior state."""
        pass


class KickoffManager(BehaviorModule):
    """
    Manages kickoff behavior with fast, optimized kickoff routine.
    Inspired by Nexto's kickoff strategy.
    """
    
    def __init__(self, enabled: bool = True, speed_boost_threshold: float = 1500):
        super().__init__(enabled)
        self.speed_boost_threshold = speed_boost_threshold
        self.kickoff_active = False
        self.kickoff_start_time = 0
    
    def should_activate(self, player: PlayerData, state: GameState) -> bool:
        """
        Activate on kickoff (ball at center, not moving much).
        """
        if not self.enabled:
            return False
        
        ball = state.ball
        # Kickoff detection: ball at origin with minimal velocity
        ball_at_center = np.linalg.norm(ball.position[:2]) < 100  # Within 100 units of center
        ball_stationary = np.linalg.norm(ball.linear_velocity) < 100
        
        # Check if we're closest to ball (simple heuristic)
        player_pos = player.car_data.position
        ball_dist = np.linalg.norm(ball.position - player_pos)
        
        # Determine if this is kickoff scenario
        is_kickoff = ball_at_center and ball_stationary
        
        if is_kickoff and not self.kickoff_active:
            self.kickoff_active = True
            self.kickoff_start_time = 0
        elif not is_kickoff:
            self.kickoff_active = False
        
        return self.kickoff_active
    
    def get_action(self, player: PlayerData, state: GameState, action_parser) -> BehaviorAction:
        """
        Execute fast kickoff: rush to ball with boost and front flip.
        """
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        player_vel = player.car_data.linear_velocity
        
        # Direction to ball
        to_ball = ball_pos - player_pos
        dist_to_ball = np.linalg.norm(to_ball)
        
        # Determine action based on distance and state
        # Action components: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        
        # Always throttle forward and boost on kickoff
        throttle = 1.0
        boost = 1 if player.boost_amount > 0 else 0
        
        # Calculate steering toward ball
        forward_dir = player.car_data.forward()
        to_ball_2d = to_ball[:2] / (np.linalg.norm(to_ball[:2]) + 1e-6)
        forward_2d = forward_dir[:2] / (np.linalg.norm(forward_dir[:2]) + 1e-6)
        
        # Cross product for steering direction
        cross = to_ball_2d[0] * forward_2d[1] - to_ball_2d[1] * forward_2d[0]
        steer = np.clip(cross * 5, -1, 1)
        
        # Jump/flip when close to ball
        jump = 1 if dist_to_ball < 300 and dist_to_ball > 150 else 0
        pitch = -1 if jump == 1 else 0
        
        # Convert to discrete action
        # Find closest matching action from action set
        desired_action = [throttle, steer, pitch, 0, 0, jump, boost, 0]
        action_idx = self._find_closest_action(desired_action, action_parser)
        
        return BehaviorAction(
            action_index=action_idx,
            confidence=0.95,
            metadata={'behavior': 'kickoff', 'dist_to_ball': dist_to_ball}
        )
    
    def _find_closest_action(self, desired_action, action_parser):
        """
        Find the closest discrete action to the desired continuous action.
        """
        if not hasattr(action_parser, 'actions'):
            return 0
        
        actions = action_parser.actions
        min_dist = float('inf')
        best_idx = 0
        
        for idx, action in enumerate(actions):
            # Calculate Euclidean distance (focus on first 3 components)
            dist = sum((desired_action[i] - action[i])**2 for i in range(min(3, len(action))))
            # Add bonus for matching boost/jump
            if len(action) > 6:
                if desired_action[6] == action[6]:
                    dist -= 0.5
                if desired_action[5] == action[5]:
                    dist -= 0.5
            
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        return best_idx
    
    def reset(self):
        """Reset kickoff state."""
        self.kickoff_active = False
        self.kickoff_start_time = 0


class RecoveryManager(BehaviorModule):
    """
    Manages recovery from aerials, flips, and tumbles.
    Helps bot quickly reorient and land on wheels.
    """
    
    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.in_recovery = False
    
    def should_activate(self, player: PlayerData, state: GameState) -> bool:
        """
        Activate when car is in air and not properly oriented.
        """
        if not self.enabled:
            return False
        
        car = player.car_data
        
        # Check if in air
        in_air = not player.on_ground and car.position[2] > 100
        
        # Check orientation (is car upside down or sideways?)
        up_vec = car.up()
        # up_vec[2] should be close to 1 if upright
        not_upright = up_vec[2] < 0.7
        
        # Check angular velocity (tumbling)
        ang_vel = car.angular_velocity
        tumbling = np.linalg.norm(ang_vel) > 1.0
        
        self.in_recovery = in_air and (not_upright or tumbling)
        
        return self.in_recovery
    
    def get_action(self, player: PlayerData, state: GameState, action_parser) -> BehaviorAction:
        """
        Execute recovery: orient car upright and prepare for landing.
        """
        car = player.car_data
        up_vec = car.up()
        
        # Goal: align up vector with world up (0, 0, 1)
        # Calculate desired pitch/yaw/roll
        
        # Simplified: pitch down to orient wheels down
        pitch = -1 if up_vec[2] < 0.9 else 0
        yaw = 0
        roll = 0
        
        # If mostly upright, use air roll to fine-tune
        if up_vec[2] > 0.5:
            # Calculate roll needed
            right_vec = car.right()
            roll = -np.sign(right_vec[2]) if abs(right_vec[2]) > 0.3 else 0
        
        throttle = 0
        steer = 0
        jump = 0
        boost = 0
        
        desired_action = [throttle, steer, pitch, yaw, roll, jump, boost, 0]
        action_idx = self._find_closest_action(desired_action, action_parser)
        
        return BehaviorAction(
            action_index=action_idx,
            confidence=0.8,
            metadata={'behavior': 'recovery', 'up_z': up_vec[2]}
        )
    
    def _find_closest_action(self, desired_action, action_parser):
        """Find closest discrete action."""
        if not hasattr(action_parser, 'actions'):
            return 0
        
        actions = action_parser.actions
        min_dist = float('inf')
        best_idx = 0
        
        for idx, action in enumerate(actions):
            dist = sum((desired_action[i] - action[i])**2 for i in range(min(5, len(action))))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        return best_idx


class BoostManager(BehaviorModule):
    """
    Manages boost collection and usage efficiency.
    Guides bot to collect boost when low and safe to do so.
    """
    
    def __init__(self, enabled: bool = True, low_boost_threshold: float = 30):
        super().__init__(enabled)
        self.low_boost_threshold = low_boost_threshold
    
    def should_activate(self, player: PlayerData, state: GameState) -> bool:
        """
        Activate when boost is low and it's safe to collect.
        """
        if not self.enabled:
            return False
        
        # Check boost level
        low_boost = player.boost_amount < self.low_boost_threshold
        
        # Check if ball is far away (safe to collect boost)
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        ball_dist = np.linalg.norm(ball_pos - player_pos)
        ball_far = ball_dist > 2000
        
        # Don't activate during critical moments
        # (ball near own goal, etc. - simplified for now)
        
        return low_boost and ball_far
    
    def get_action(self, player: PlayerData, state: GameState, action_parser) -> BehaviorAction:
        """
        Navigate toward nearest boost pad.
        Note: This is a simplified version. In practice, we'd need
        boost pad locations from game state.
        """
        # For now, return None to let policy handle it
        # Full implementation would require boost pad state tracking
        return BehaviorAction(action_index=None, confidence=0.0)


class BehaviorCoordinator:
    """
    Coordinates multiple behavior modules and decides which should take control.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.behaviors = []
        
        # Initialize behavior modules
        behavior_config = self.config.get('behaviors', {})
        
        self.kickoff_manager = KickoffManager(
            enabled=behavior_config.get('kickoff_enabled', True)
        )
        self.recovery_manager = RecoveryManager(
            enabled=behavior_config.get('recovery_enabled', True)
        )
        self.boost_manager = BoostManager(
            enabled=behavior_config.get('boost_management_enabled', False)
        )
        
        # Priority order (first to activate wins)
        self.behaviors = [
            self.kickoff_manager,
            self.recovery_manager,
            self.boost_manager,
        ]
    
    def get_behavior_action(
        self,
        player: PlayerData,
        state: GameState,
        action_parser
    ) -> Optional[BehaviorAction]:
        """
        Check all behaviors and return action from highest priority active behavior.
        
        Args:
            player: Current player data
            state: Current game state
            action_parser: Action parser for discrete actions
            
        Returns:
            BehaviorAction if any behavior wants to override, None otherwise
        """
        for behavior in self.behaviors:
            if behavior.should_activate(player, state):
                return behavior.get_action(player, state, action_parser)
        
        return None
    
    def reset(self):
        """Reset all behaviors."""
        for behavior in self.behaviors:
            behavior.reset()
