import numpy as np
from typing import List

from sequences.sequence import Sequence
from util.game_state import GameState, PlayerData

class WaveDash(Sequence):
    def __init__(self):
        self.step = 0
        self.jump_time = 0.0
        self.dodge_time = 0.0
        self.duration = 0.15 # Total duration of the dodge part
        self.jump_duration = 0.05 # Duration of the initial jump

    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        # Check if the car is on the ground or wall/ceiling and has a flip available
        # We'll simplify and assume a ground/wall dash for now.
        # A full implementation would check for a specific angle to the ground/wall.
        
        # Check if car is on the ground or close to it (z-axis)
        on_ground = player.on_ground

        # Check if car has a flip available and isn't currently mid-flip
        airborne_without_flip = player.jumped and not player.on_ground
        has_flip = player.has_flip and not airborne_without_flip
        
        # Check if the car is moving fast enough to make a wavedash useful
        speed = np.linalg.norm(player.car_data.linear_velocity)
        is_fast_enough = speed > 500
        
        # Only allow a wavedash if we are on the ground/wall and have a flip
        return (on_ground or player.is_on_wall) and has_flip and is_fast_enough

    def get_action(self, player: PlayerData, game_state: GameState, prev_action: np.ndarray) -> List:
        # Reset sequence if it's the first step
        if self.step == 0:
            self.jump_time = game_state.time
            self.step = 1
            
        controls = prev_action.copy()
        
        # Step 1: Jump
        if self.step == 1:
            controls[5] = 1.0 # Jump
            
            if game_state.time - self.jump_time >= self.jump_duration:
                self.dodge_time = game_state.time
                self.step = 2
        
        # Step 2: Dodge (forward or backward)
        elif self.step == 2:
            controls[5] = 0.0 # Release jump
            
            # Determine dodge direction based on car's orientation and velocity
            # For simplicity, we'll assume a forward dash for now.
            # A proper implementation would use the car's orientation vector.
            
            # Forward dodge: pitch = -1.0
            controls[2] = -1.0 # Pitch
            controls[5] = 1.0 # Jump (for the dodge)
            
            if game_state.time - self.dodge_time >= self.duration:
                self.step = 3
                
        # Step 3: Finish
        elif self.step == 3:
            # Release all controls related to the dodge
            controls[2] = 0.0 # Pitch
            controls[5] = 0.0 # Jump
            
            # Sequence is complete, reset for next time
            self.step = 0
            
        return controls.tolist()

    def is_finished(self) -> bool:
        return self.step == 0

    def reset(self):
        self.step = 0
        self.jump_time = 0.0
        self.dodge_time = 0.0
