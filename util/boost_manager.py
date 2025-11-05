"""
Boost Management System
Tracks boost pad states, manages boost collection strategy, and implements boost denial.
Critical for SSL-level play - boost advantage wins 1v1 games.
"""

import numpy as np
from typing import List, Optional, Tuple
from .game_state import GameState
from .player_data import PlayerData


class BoostPad:
    """Represents a single boost pad on the field"""
    def __init__(self, index: int, position: np.ndarray, is_big: bool):
        self.index = index
        self.position = position
        self.is_big = is_big
        self.is_active = True
        self.respawn_time = 10.0 if is_big else 4.0
        self.time_since_pickup = 0.0
        self.value = 100 if is_big else 12  # Boost amount provided
    
    def update(self, dt: float, is_active: bool):
        """Update pad state and timer"""
        if not is_active and not self.is_active:
            # Pad is respawning, update timer
            self.time_since_pickup += dt
            if self.time_since_pickup >= self.respawn_time:
                self.is_active = True
                self.time_since_pickup = 0.0
        elif not is_active and self.is_active:
            # Pad was just picked up
            self.is_active = False
            self.time_since_pickup = 0.0
        elif is_active:
            # Pad is active
            self.is_active = True
            self.time_since_pickup = 0.0
    
    def time_until_respawn(self) -> float:
        """Time in seconds until pad respawns"""
        if self.is_active:
            return 0.0
        return max(0.0, self.respawn_time - self.time_since_pickup)


class BoostManager:
    """
    Manages boost collection strategy and boost pad tracking.
    Implements boost stealing and boost conservation logic.
    """
    
    # Standard Rocket League boost pad positions (approximated)
    # In a real implementation, these would be loaded from field info
    BIG_PAD_POSITIONS = [
        np.array([3072, 4096, 73]),    # Orange back right
        np.array([-3072, 4096, 73]),   # Orange back left
        np.array([3072, -4096, 73]),   # Blue back right
        np.array([-3072, -4096, 73]),  # Blue back left
        np.array([3584, 0, 73]),       # Right mid
        np.array([-3584, 0, 73]),      # Left mid
    ]
    
    def __init__(self, field_info=None):
        """
        Initialize boost manager.
        
        Args:
            field_info: FieldInfoPacket from RLBot (contains boost pad positions)
        """
        self.pads: List[BoostPad] = []
        self.last_update_time = 0.0
        
        # Initialize boost pads
        # In real implementation, would use field_info.boost_pads
        # For now, we'll use a simplified setup
        self._initialize_default_pads()
    
    def _initialize_default_pads(self):
        """Initialize with default boost pad layout"""
        # Add 6 big pads
        for i, pos in enumerate(self.BIG_PAD_POSITIONS):
            self.pads.append(BoostPad(i, pos, is_big=True))
        
        # Note: In real implementation, would add all 34 small pads too
        # Skipping for brevity, but the logic would be identical
    
    def update(self, game_state: GameState, dt: float):
        """
        Update boost pad states based on game state.
        
        Args:
            game_state: Current game state
            dt: Time elapsed since last update
        """
        # Update each pad based on game_state.boost_pads
        for i, pad in enumerate(self.pads):
            if i < len(game_state.boost_pads):
                is_active = game_state.boost_pads[i] > 0.5
                pad.update(dt, is_active)
    
    def get_best_boost_to_collect(self, player: PlayerData, opponent: PlayerData,
                                   game_state: GameState) -> Optional[BoostPad]:
        """
        Determine which boost pad to collect based on strategy.
        
        Strategy considerations:
        - Distance to pad
        - Pad respawn timer
        - Current boost amount
        - Opponent position (boost stealing opportunity)
        - Game situation (offensive vs defensive)
        
        Args:
            player: Our player state
            opponent: Opponent player state
            game_state: Current game state
            
        Returns:
            Best BoostPad to collect, or None if shouldn't collect boost
        """
        current_boost = player.boost_amount
        
        # Don't collect if we have plenty of boost
        if current_boost > 0.75:
            return None
        
        car_pos = player.car_data.position
        opponent_pos = opponent.car_data.position
        
        best_pad = None
        best_score = -float('inf')
        
        for pad in self.pads:
            if not pad.is_active and pad.time_until_respawn() > 1.0:
                continue  # Skip pads that won't respawn soon
            
            score = self._calculate_pad_value(pad, car_pos, opponent_pos, current_boost)
            
            if score > best_score:
                best_score = score
                best_pad = pad
        
        return best_pad if best_score > 0 else None
    
    def _calculate_pad_value(self, pad: BoostPad, car_pos: np.ndarray,
                            opponent_pos: np.ndarray, current_boost: float) -> float:
        """
        Calculate utility value of collecting a specific boost pad.
        
        Returns:
            Score (higher = more valuable to collect)
        """
        score = 0.0
        
        # Base value from boost amount
        score += pad.value * 0.5
        
        # Distance penalty (closer is better)
        distance_to_pad = np.linalg.norm(car_pos - pad.position)
        score -= distance_to_pad * 0.01  # Penalize distance
        
        # Respawn timer penalty
        if not pad.is_active:
            score -= pad.time_until_respawn() * 5.0
        
        # Boost stealing bonus (if opponent is close to this pad)
        opponent_distance = np.linalg.norm(opponent_pos - pad.position)
        if opponent_distance < distance_to_pad * 1.5:
            # Opponent might be going for this boost
            score += 20.0  # Bonus for denying opponent
        
        # Urgency multiplier (lower boost = more urgent)
        urgency = 1.0 + (1.0 - current_boost) * 2.0
        score *= urgency
        
        # Big pad bonus (big pads are always valuable)
        if pad.is_big:
            score *= 2.0
        
        return score
    
    def should_steal_boost(self, player: PlayerData, opponent: PlayerData,
                          game_state: GameState) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Decide if we should go for opponent's boost (boost denial strategy).
        
        This is a key SSL tactic: take opponent's corner boost to starve them.
        
        Args:
            player: Our player state
            opponent: Opponent player state
            game_state: Current game state
            
        Returns:
            (should_steal, target_position) where target is opponent's corner boost
        """
        car_pos = player.car_data.position
        opponent_pos = opponent.car_data.position
        ball_pos = game_state.ball.position
        
        # Determine which team we are
        our_goal_y = -5120 if player.team_num == 0 else 5120
        opponent_goal_y = 5120 if player.team_num == 0 else -5120
        
        # Find opponent's back corner boosts
        opponent_corner_boosts = []
        for pad in self.pads:
            if not pad.is_big:
                continue
            
            # Check if pad is in opponent's half
            pad_y = pad.position[1]
            if (player.team_num == 0 and pad_y > 0) or (player.team_num == 1 and pad_y < 0):
                opponent_corner_boosts.append(pad)
        
        if not opponent_corner_boosts:
            return False, None
        
        # Decide if boost steal is good strategy right now
        should_steal = False
        
        # Condition 1: We have ball control or ball is in opponent's half
        ball_in_opponent_half = (
            (player.team_num == 0 and ball_pos[1] > 0) or
            (player.team_num == 1 and ball_pos[1] < 0)
        )
        
        # Condition 2: Opponent is low on boost
        opponent_low_boost = opponent.boost_amount < 0.3
        
        # Condition 3: We have enough boost to make the play
        we_have_boost = player.boost_amount > 0.5
        
        # Condition 4: We're in offensive position
        we_offensive = (
            (player.team_num == 0 and car_pos[1] > ball_pos[1]) or
            (player.team_num == 1 and car_pos[1] < ball_pos[1])
        )
        
        # Decide based on conditions
        if ball_in_opponent_half and we_have_boost and (opponent_low_boost or we_offensive):
            should_steal = True
            
            # Pick closest opponent corner boost
            target_boost = min(opponent_corner_boosts, 
                             key=lambda p: np.linalg.norm(p.position - car_pos))
            
            return True, target_boost.position
        
        return False, None
    
    def get_boost_conservation_mode(self, player: PlayerData, 
                                   game_state: GameState) -> bool:
        """
        Decide if we should conserve boost (use it sparingly).
        
        Returns:
            True if should conserve boost, False if can use freely
        """
        current_boost = player.boost_amount
        
        # Conserve if low on boost
        if current_boost < 0.3:
            return True
        
        # Check if we have access to boost pads
        car_pos = player.car_data.position
        
        # Find closest active boost pad
        closest_pad_dist = float('inf')
        for pad in self.pads:
            if pad.is_active:
                dist = np.linalg.norm(car_pos - pad.position)
                closest_pad_dist = min(closest_pad_dist, dist)
        
        # Conserve if no boost pads nearby
        if closest_pad_dist > 2000:
            return True
        
        return False
    
    def get_boost_path(self, player: PlayerData, target_position: np.ndarray,
                       game_state: GameState) -> List[np.ndarray]:
        """
        Calculate optimal path to target that collects boost pads along the way.
        
        Args:
            player: Our player state
            target_position: Where we want to go
            game_state: Current game state
            
        Returns:
            List of waypoints including boost pads to collect
        """
        car_pos = player.car_data.position
        path = [car_pos.copy()]
        
        # Find pads along the path to target
        direction = target_position - car_pos
        distance = np.linalg.norm(direction)
        direction_norm = direction / distance if distance > 0 else np.zeros(3)
        
        # Check each pad to see if it's roughly on our path
        pads_on_path = []
        for pad in self.pads:
            if not pad.is_active:
                continue
            
            # Vector from car to pad
            car_to_pad = pad.position - car_pos
            
            # Dot product to check if pad is in our direction
            alignment = np.dot(car_to_pad, direction_norm)
            
            # Check if pad is along our path
            if alignment > 0:
                # Distance from pad to our straight-line path
                pad_dist_from_path = np.linalg.norm(car_to_pad - alignment * direction_norm)
                
                # If pad is close to our path, add it
                if pad_dist_from_path < 500:  # Within 500 units of path
                    pads_on_path.append((pad, alignment))
        
        # Sort pads by distance along path
        pads_on_path.sort(key=lambda x: x[1])
        
        # Add pads to path (limit to 2 pads to avoid over-correction)
        for pad, _ in pads_on_path[:2]:
            path.append(pad.position)
        
        # Add final target
        path.append(target_position)
        
        return path
    
    def get_mid_boost_control_score(self, player: PlayerData, 
                                   opponent: PlayerData) -> float:
        """
        Calculate how well we control mid boost (center pads).
        SSL players fight for mid boost control.
        
        Returns:
            Score from -1 (opponent controls) to +1 (we control)
        """
        car_pos = player.car_data.position
        opponent_pos = opponent.car_data.position
        
        # Find mid boost pads (near x=0, y=0)
        mid_pads = [pad for pad in self.pads if pad.is_big and abs(pad.position[1]) < 1000]
        
        if not mid_pads:
            return 0.0
        
        our_control = 0.0
        opponent_control = 0.0
        
        for pad in mid_pads:
            our_dist = np.linalg.norm(car_pos - pad.position)
            opp_dist = np.linalg.norm(opponent_pos - pad.position)
            
            if our_dist < opp_dist:
                our_control += 1.0
            else:
                opponent_control += 1.0
        
        total = our_control + opponent_control
        return (our_control - opponent_control) / total if total > 0 else 0.0
