"""
Bumping and Demo Mechanic
Executes strategic bumps and demolitions to disrupt opponents.
Critical for offensive pressure and defensive disruption.
"""

import numpy as np
from typing import List, Optional
from util.player_data import PlayerData
from util.game_state import GameState


class BumpMechanic:
    """
    Strategic bumping: Position for demolition or disruption bump.
    Tracks opponent positions and speeds to execute demos.
    """
    
    # Demo requires 2300+ uu/s relative speed
    DEMO_SPEED_THRESHOLD = 2300.0
    # Minimum boost for demo attempt
    MIN_BOOST_FOR_DEMO = 20.0
    
    def __init__(self, target_player_id: Optional[int] = None):
        """Initialize bump mechanic.
        
        Args:
            target_player_id: Specific player to target (None = auto-select)
        """
        self.target_player_id = target_player_id
        self.state = 'idle'
        self.approach_start_time = 0.0
        
    def select_target(self, player: PlayerData, game_state: GameState) -> Optional[int]:
        """Select best opponent to bump/demo.
        
        Prioritizes:
        1. Opponents close to ball
        2. Opponents in defensive position
        3. Closest opponent
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            Target player ID or None
        """
        opponents = game_state.get_opponents(player.team)
        if not opponents:
            return None
        
        ball_pos = game_state.ball.position
        player_pos = player.car_data.position
        
        # Score each opponent
        best_score = -float('inf')
        best_opponent = None
        
        for opp in opponents:
            opp_pos = opp.car_data.position
            
            # Distance to ball (closer is better target)
            dist_to_ball = np.linalg.norm(opp_pos - ball_pos)
            ball_proximity_score = 1.0 / (dist_to_ball + 1.0)
            
            # Distance from us (closer is easier to reach)
            dist_from_us = np.linalg.norm(opp_pos - player_pos)
            reachability_score = 1.0 / (dist_from_us + 1.0)
            
            # Opponent in defensive position (high priority)
            opp_team_goal = game_state.get_goal_position(opp.team)
            dist_to_own_goal = np.linalg.norm(opp_pos - opp_team_goal)
            defensive_score = 1.0 / (dist_to_own_goal + 1000.0)
            
            # Combined score
            score = (
                2.0 * ball_proximity_score +
                1.0 * reachability_score +
                3.0 * defensive_score
            )
            
            if score > best_score:
                best_score = score
                best_opponent = opp.player_id
        
        return best_opponent
    
    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if bump can be executed.
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            True if bump is feasible
        """
        # Need boost for speed
        has_boost = player.boost_amount > self.MIN_BOOST_FOR_DEMO
        
        # Should be on ground (easier to control)
        on_ground = player.on_ground
        
        # Need opponents
        opponents = game_state.get_opponents(player.team)
        has_targets = len(opponents) > 0
        
        return has_boost and on_ground and has_targets
    
    def calculate_demo_feasibility(
        self, 
        player: PlayerData, 
        opponent: PlayerData
    ) -> float:
        """Calculate if demo is possible.
        
        Args:
            player: Current player
            opponent: Target opponent
            
        Returns:
            Demo feasibility score (0-1)
        """
        # Calculate relative velocity
        player_vel = player.car_data.linear_velocity
        opp_vel = opponent.car_data.linear_velocity
        relative_vel = player_vel - opp_vel
        relative_speed = np.linalg.norm(relative_vel)
        
        # Speed requirement
        if relative_speed < self.DEMO_SPEED_THRESHOLD:
            speed_score = relative_speed / self.DEMO_SPEED_THRESHOLD
        else:
            speed_score = 1.0
        
        # Supersonic bonus
        is_supersonic = np.linalg.norm(player_vel) >= 2300.0
        supersonic_bonus = 0.3 if is_supersonic else 0.0
        
        # Direction alignment (heading toward opponent)
        player_pos = player.car_data.position
        opp_pos = opponent.car_data.position
        to_opp = opp_pos - player_pos
        to_opp_norm = to_opp / (np.linalg.norm(to_opp) + 1e-8)
        
        player_forward = player.car_data.forward()
        alignment = np.dot(player_forward, to_opp_norm)
        alignment_score = max(0.0, alignment)  # 0 to 1
        
        # Combined score
        feasibility = (
            0.5 * speed_score +
            0.3 * alignment_score +
            0.2 * supersonic_bonus
        )
        
        return min(1.0, feasibility)
    
    def execute(
        self, 
        player: PlayerData, 
        game_state: GameState,
        prev_action: np.ndarray
    ) -> List[float]:
        """Execute bump/demo sequence.
        
        Args:
            player: Current player
            game_state: Current game state  
            prev_action: Previous action
            
        Returns:
            Action controls [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        # Select target if not set
        if self.target_player_id is None:
            self.target_player_id = self.select_target(player, game_state)
        
        if self.target_player_id is None:
            # No valid target
            return prev_action.tolist()
        
        # Get target opponent
        target_opp = game_state.get_player_by_id(self.target_player_id)
        if target_opp is None:
            self.target_player_id = None
            return prev_action.tolist()
        
        # Calculate approach
        player_pos = player.car_data.position
        opp_pos = target_opp.car_data.position
        opp_vel = target_opp.car_data.linear_velocity
        
        # Lead target based on opponent velocity
        # Predict where opponent will be
        time_to_intercept = np.linalg.norm(opp_pos - player_pos) / 2300.0  # Assume supersonic
        predicted_pos = opp_pos + opp_vel * time_to_intercept
        
        # Direction to target
        to_target = predicted_pos - player_pos
        distance = np.linalg.norm(to_target)
        
        if distance < 50:
            # Very close, just drive straight
            to_target_norm = to_target / (distance + 1e-8)
        else:
            to_target_norm = to_target / distance
        
        # Calculate steering
        player_forward = player.car_data.forward()
        player_right = player.car_data.right()
        
        # Steering angle
        steer_angle = np.dot(to_target_norm, player_right)
        steer = np.clip(steer_angle * 2.0, -1.0, 1.0)
        
        # Throttle: full speed ahead
        throttle = 1.0
        
        # Boost: use if aligned and need speed
        alignment = np.dot(player_forward, to_target_norm)
        player_speed = np.linalg.norm(player.car_data.linear_velocity)
        
        should_boost = (
            alignment > 0.8 and 
            player_speed < 2200.0 and 
            player.boost_amount > 10.0
        )
        boost = 1.0 if should_boost else 0.0
        
        # Handbrake: use for sharp turns
        sharp_turn = abs(steer) > 0.8 and player_speed > 1000.0
        handbrake = 1.0 if sharp_turn else 0.0
        
        return [throttle, steer, 0.0, 0.0, 0.0, 0.0, boost, handbrake]
    
    def is_finished(self) -> bool:
        """Check if bump sequence is complete."""
        # Bump is never "finished" - it's a continuous behavior
        return False
    
    def reset(self):
        """Reset mechanic state."""
        self.state = 'idle'
        self.target_player_id = None
        self.approach_start_time = 0.0
