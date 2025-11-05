import numpy as np
from typing import List
from util.game_state import GameState, PlayerData
from sequences.sequence import Sequence

class WaveDash(Sequence):
    """
    WaveDash mechanic: Jump + immediate forward dodge while pitching back
    Provides quick speed boost (~500 uu/s) when executed correctly.
    Fixed version: Uses state-based logic instead of time-based.
    """
    def __init__(self):
        self.state = 'idle'
        self.initial_height = 0.0
        self.jump_registered = False

    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if wavedash can be executed"""
        # Must be in idle state (not currently executing)
        if self.state != 'idle':
            return False
        
        # Must be on ground
        on_ground = player.on_ground
        
        # Must have flip available
        has_flip = player.has_flip
        
        # Must be moving at moderate speed (not too slow, not at max speed)
        speed = np.linalg.norm(player.car_data.linear_velocity)
        good_speed = 500 < speed < 1800
        
        # Must not be already airborne
        low_height = player.car_data.position[2] < 50
        
        return on_ground and has_flip and good_speed and low_height

    def get_action(self, player: PlayerData, game_state: GameState, prev_action: np.ndarray) -> List:
        """Execute wavedash sequence based on physics state"""
        car = player.car_data
        
        if self.state == 'idle':
            # Start sequence - initiate jump
            self.state = 'jumping'
            self.initial_height = car.position[2]
            self.jump_registered = False
            # Jump + boost + pitch back slightly
            return [1.0, 0.0, 0.3, 0.0, 0.0, 1.0, 1.0, 0.0]
        
        elif self.state == 'jumping':
            # Wait for car to leave ground
            height_gain = car.position[2] - self.initial_height
            
            if height_gain > 10 or not player.on_ground:
                # Car is airborne, now dodge forward
                self.state = 'dodging'
                self.jump_registered = True
                # Forward dodge with pitch back (creates wavedash)
                return [1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0]
            else:
                # Still waiting to leave ground, hold jump
                return [1.0, 0.0, 0.3, 0.0, 0.0, 1.0, 1.0, 0.0]
        
        elif self.state == 'dodging':
            # Execute dodge - forward flip with pitch compensation
            if not player.on_ground and self.jump_registered:
                # In air, do the dodge
                return [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            elif player.on_ground:
                # Landed - sequence complete
                self.state = 'idle'
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                # Still in air
                return [1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        # Fallback - should not reach here
        self.state = 'idle'
        return prev_action.tolist()

    def is_finished(self) -> bool:
        """Check if sequence is complete"""
        return self.state == 'idle'

    def reset(self):
        """Reset sequence to initial state"""
        self.state = 'idle'
        self.initial_height = 0.0
        self.jump_registered = False
