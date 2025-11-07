"""
Recovery Mechanic
Handles landing recovery and reorientation after aerials.
Critical for maintaining momentum and control flow.
"""

import numpy as np
from typing import List, Optional
from util.player_data import PlayerData
from util.game_state import GameState


class RecoveryMechanic:
    """
    Recovery: Reorient car to land wheels-down and maintain momentum.
    Essential after aerials, bumps, and awkward situations.
    """
    
    STATE_IDLE = 'idle'
    STATE_ORIENT = 'orient'
    STATE_WAVEDASH = 'wavedash'
    STATE_COMPLETE = 'complete'
    
    def __init__(self):
        """Initialize recovery mechanic."""
        self.state = self.STATE_IDLE
        self.target_orientation = None
        self.wavedash_possible = False
        
    def is_valid(self, player: PlayerData) -> bool:
        """Check if recovery is needed.
        
        Args:
            player: Current player
            
        Returns:
            True if recovery should be performed
        """
        # Not on ground
        not_grounded = not player.on_ground
        
        # Car is misaligned (not wheels-down)
        car_up = player.car_data.up()
        misaligned = car_up[2] < 0.8  # Up vector not pointing up enough
        
        # Close to ground (recovery needed soon)
        car_pos = player.car_data.position
        low_altitude = car_pos[2] < 500  # Below 500 units
        
        return not_grounded and (misaligned or low_altitude)
    
    def calculate_target_orientation(
        self,
        player: PlayerData,
        game_state: GameState
    ) -> np.ndarray:
        """Calculate desired landing orientation.
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            Target up vector (usually [0, 0, 1] for wheels down)
        """
        # Default: wheels down
        target_up = np.array([0, 0, 1])
        
        # If on wall, orient to wall
        car_pos = player.car_data.position
        
        # Field dimensions
        FIELD_WIDTH = 8192
        FIELD_LENGTH = 10240
        
        # Check if near wall
        near_left_wall = abs(car_pos[0] + FIELD_WIDTH / 2) < 300
        near_right_wall = abs(car_pos[0] - FIELD_WIDTH / 2) < 300
        near_back_wall = abs(car_pos[1] + FIELD_LENGTH / 2) < 300
        near_front_wall = abs(car_pos[1] - FIELD_LENGTH / 2) < 300
        
        if near_left_wall:
            target_up = np.array([-1, 0, 0])
        elif near_right_wall:
            target_up = np.array([1, 0, 0])
        elif near_back_wall:
            target_up = np.array([0, -1, 0])
        elif near_front_wall:
            target_up = np.array([0, 1, 0])
        
        return target_up
    
    def can_wavedash(self, player: PlayerData) -> bool:
        """Check if wavedash recovery is possible.
        
        Wavedash: Jump just before landing + immediate dodge
        Maintains/boosts speed on landing
        
        Args:
            player: Current player
            
        Returns:
            True if wavedash is possible
        """
        # Must have flip
        has_flip = player.has_flip
        
        # Must be close to ground
        car_pos = player.car_data.position
        close_to_ground = 50 < car_pos[2] < 150  # Sweet spot for wavedash
        
        # Must be oriented somewhat correctly (not upside down)
        car_up = player.car_data.up()
        oriented = car_up[2] > 0.3
        
        # Must have forward velocity
        car_vel = player.car_data.linear_velocity
        has_momentum = np.linalg.norm(car_vel[:2]) > 500  # Horizontal speed
        
        return has_flip and close_to_ground and oriented and has_momentum
    
    def execute(
        self,
        player: PlayerData,
        game_state: GameState,
        prev_action: np.ndarray
    ) -> List[float]:
        """Execute recovery sequence.
        
        Args:
            player: Current player
            game_state: Current game state
            prev_action: Previous action
            
        Returns:
            Action controls
        """
        if self.state == self.STATE_IDLE:
            # Start recovery
            self.target_orientation = self.calculate_target_orientation(player, game_state)
            self.wavedash_possible = self.can_wavedash(player)
            
            if self.wavedash_possible:
                self.state = self.STATE_WAVEDASH
            else:
                self.state = self.STATE_ORIENT
        
        if self.state == self.STATE_ORIENT:
            # Reorient car to target orientation
            controls = self._calculate_orientation_controls(
                player.car_data,
                self.target_orientation
            )
            
            # Check if oriented correctly
            car_up = player.car_data.up()
            alignment = np.dot(car_up, self.target_orientation)
            
            if alignment > 0.95:
                # Well oriented
                self.state = self.STATE_COMPLETE
            
            return controls
        
        elif self.state == self.STATE_WAVEDASH:
            # Execute wavedash
            car_pos = player.car_data.position
            
            # Jump just before ground contact
            if 80 < car_pos[2] < 120:
                # Jump
                return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif car_pos[2] < 80:
                # Dodge forward for speed boost
                return [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            else:
                # Orient while falling
                controls = self._calculate_orientation_controls(
                    player.car_data,
                    self.target_orientation
                )
                return controls
        
        elif self.state == self.STATE_COMPLETE:
            # Recovery complete
            self.state = self.STATE_IDLE
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return prev_action.tolist()
    
    def _calculate_orientation_controls(
        self,
        car,
        target_up: np.ndarray
    ) -> List[float]:
        """Calculate controls to orient car.
        
        Args:
            car: Car data
            target_up: Target up vector
            
        Returns:
            Controls list [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        current_up = car.up()
        
        # Calculate rotation needed
        # Cross product gives axis of rotation
        rotation_axis = np.cross(current_up, target_up)
        rotation_magnitude = np.linalg.norm(rotation_axis)
        
        if rotation_magnitude < 0.01:
            # Already aligned
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        rotation_axis_norm = rotation_axis / rotation_magnitude
        
        # Map rotation axis to pitch/yaw/roll controls
        # This is simplified - proper implementation uses rotation matrices
        
        # Get car orientation
        forward = car.forward()
        right = car.right()
        
        # Project rotation axis onto car axes
        pitch_component = -np.dot(rotation_axis_norm, right)
        yaw_component = np.dot(rotation_axis_norm, forward)
        roll_component = np.dot(rotation_axis_norm, current_up)
        
        # Scale by rotation magnitude for proportional control
        gain = 2.0
        pitch = np.clip(pitch_component * gain, -1, 1)
        yaw = np.clip(yaw_component * gain, -1, 1)
        roll = np.clip(roll_component * gain, -1, 1)
        
        # Boost if aligned and need to gain speed
        car_vel = car.linear_velocity
        speed = np.linalg.norm(car_vel)
        boost = 0.5 if speed < 1400 and current_up[2] > 0.7 else 0.0
        
        return [1.0, 0.0, pitch, yaw, roll, 0.0, boost, 0.0]
    
    def is_finished(self) -> bool:
        """Check if recovery is complete."""
        return self.state == self.STATE_IDLE or self.state == self.STATE_COMPLETE
    
    def reset(self):
        """Reset recovery state."""
        self.state = self.STATE_IDLE
        self.target_orientation = None
        self.wavedash_possible = False


class HalfFlipRecovery(RecoveryMechanic):
    """
    Half-Flip: Quick 180-degree turn while maintaining momentum.
    Used for fast direction changes and recoveries.
    """
    
    STATE_BACKFLIP = 'backflip'
    STATE_CANCEL = 'cancel'
    STATE_ROLL = 'roll'
    
    def __init__(self):
        super().__init__()
        self.flip_time = 0.0
        
    def is_valid(self, player: PlayerData) -> bool:
        """Check if half-flip is appropriate."""
        # Must be on ground
        on_ground = player.on_ground
        
        # Must have flip
        has_flip = player.has_flip
        
        # Should have backward momentum or need to turn around
        # This is simplified - real logic would check if turning around is beneficial
        
        return on_ground and has_flip
    
    def execute(
        self,
        player: PlayerData,
        game_state: GameState,
        prev_action: np.ndarray,
        delta_time: float = 1/120.0
    ) -> List[float]:
        """Execute half-flip sequence."""
        if self.state == self.STATE_IDLE:
            self.state = self.STATE_BACKFLIP
            self.flip_time = 0.0
        
        if self.state == self.STATE_BACKFLIP:
            # Back flip
            self.flip_time += delta_time
            
            if self.flip_time < 0.1:
                # Jump + backflip
                return [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            else:
                self.state = self.STATE_CANCEL
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif self.state == self.STATE_CANCEL:
            # Cancel flip rotation with forward pitch
            self.flip_time += delta_time
            
            if self.flip_time < 0.3:
                return [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                self.state = self.STATE_ROLL
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif self.state == self.STATE_ROLL:
            # Air roll to land correctly
            self.flip_time += delta_time
            
            if self.flip_time < 0.5:
                # Determine roll direction based on current orientation
                car_right = player.car_data.right()
                roll_dir = 1.0 if car_right[2] > 0 else -1.0
                return [1.0, 0.0, 0.0, 0.0, roll_dir, 0.0, 0.0, 0.0]
            else:
                self.state = self.STATE_COMPLETE
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return super().execute(player, game_state, prev_action)
