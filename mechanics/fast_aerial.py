"""
Fast Aerial Mechanic
Executes fast aerial takeoff for quicker ball interception.
Critical for winning 50/50s and beating opponent to aerial balls.
"""

import numpy as np
from typing import List, Optional
from util.player_data import PlayerData
from util.game_state import GameState


class FastAerial:
    """
    Fast aerial: Jump + tilt back + dodge + boost
    Provides ~50% faster aerial takeoff than regular aerial.
    """
    
    def __init__(self, target_position: np.ndarray):
        self.target = target_position
        self.state = 'idle'
        self.initial_height = 0.0
        self.jump_initiated = False
        
    def is_valid(self, player: PlayerData) -> bool:
        """Check if fast aerial can be executed"""
        # Must be on ground
        on_ground = player.on_ground
        
        # Must have flip
        has_flip = player.has_flip
        
        # Must have boost
        has_boost = player.boost_amount > 0.3
        
        # Must be in idle state
        idle = self.state == 'idle'
        
        return on_ground and has_flip and has_boost and idle
    
    def execute(self, player: PlayerData, game_state: GameState,
                prev_action: np.ndarray) -> List[float]:
        """Execute fast aerial sequence"""
        car = player.car_data
        
        if self.state == 'idle':
            # Start sequence
            self.state = 'first_jump'
            self.initial_height = car.position[2]
            self.jump_initiated = False
            
        if self.state == 'first_jump':
            # Jump + tilt back + boost
            if not self.jump_initiated:
                self.jump_initiated = True
                return [1.0, 0.0, -0.8, 0.0, 0.0, 1.0, 1.0, 0.0]  # Jump + pitch back + boost
            
            # Wait for car to start leaving ground
            if car.position[2] > self.initial_height + 10 or not player.on_ground:
                self.state = 'dodge'
                return [1.0, 0.0, -0.8, 0.0, 0.0, 0.0, 1.0, 0.0]  # Release jump
            else:
                return [1.0, 0.0, -0.8, 0.0, 0.0, 1.0, 1.0, 0.0]  # Hold jump + boost
        
        elif self.state == 'dodge':
            # Execute dodge for extra acceleration
            if not player.on_ground:
                self.state = 'aerial_control'
                return [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0]  # Dodge + boost
            else:
                return [1.0, 0.0, -0.8, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        elif self.state == 'aerial_control':
            # Point toward target and boost
            controls = self._calculate_aerial_controls(car, self.target)
            
            # Check if we've reached target or should transition
            distance_to_target = np.linalg.norm(car.position - self.target)
            if distance_to_target < 200 or player.on_ground:
                self.state = 'idle'
                return controls
            
            return controls
        
        # Fallback
        return prev_action.tolist()
    
    def _calculate_aerial_controls(self, car, target: np.ndarray) -> List[float]:
        """Calculate controls to point car toward target in air"""
        # Direction to target
        to_target = target - car.position
        distance = np.linalg.norm(to_target)
        
        if distance < 1:
            # Already at target
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        to_target_norm = to_target / distance
        
        # Current forward direction
        forward = car.forward()
        
        # Calculate pitch, yaw, roll to point at target
        # This is simplified - a full implementation would use rotation matrices
        
        # Pitch: angle between forward and target in xz plane
        forward_xz = np.array([forward[0], forward[1], 0])
        target_xz = np.array([to_target_norm[0], to_target_norm[1], 0])
        
        # Simple PID-like control
        pitch = -to_target_norm[2] * 1.5  # Negative because pitch down is positive
        pitch = np.clip(pitch, -1, 1)
        
        # Yaw: angle in xy plane - use proper proportional control
        # Calculate angle difference and convert to control input
        yaw_angle = np.arctan2(to_target_norm[1] - forward[1], to_target_norm[0] - forward[0])
        # Apply proportional control with gain
        yaw = yaw_angle * 0.5  # Proportional control with gain of 0.5
        yaw = np.clip(yaw, -1, 1)
        
        # Boost if pointing roughly toward target
        alignment = np.dot(forward, to_target_norm)
        boost = 1.0 if alignment > 0.8 and distance > 500 else 0.0
        
        return [1.0, 0.0, pitch, yaw, 0.0, 0.0, boost, 0.0]
    
    def is_finished(self) -> bool:
        """Check if sequence is complete"""
        return self.state == 'idle'
    
    def reset(self):
        """Reset to idle state"""
        self.state = 'idle'
        self.initial_height = 0.0
        self.jump_initiated = False
