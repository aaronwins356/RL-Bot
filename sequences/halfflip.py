import numpy as np
from typing import List
from util.game_state import GameState, PlayerData
from sequences.sequence import Sequence

class HalfFlip(Sequence):
    def __init__(self):
        self.step = 0
        self.flip_time = 0.0
        self.cancel_time = 0.0
        self.roll_time = 0.0
        self.flip_duration = 0.15
        self.cancel_duration = 0.05
        self.roll_duration = 0.5

    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        # A half-flip is useful when the car is on the ground and needs to quickly turn around.
        # Check if the car is on the ground and moving backward.
        on_ground = player.car_data.position[2] < 50
        speed = np.linalg.norm(player.car_data.linear_velocity)
        
        # Check if the car is moving backward (dot product of velocity and forward vector is negative)
        # For simplicity, we'll just check if speed is low and the car is on the ground.
        # A proper implementation would check the direction of travel.
        is_moving_slowly = speed < 500
        
        return on_ground and is_moving_slowly and player.car_data.has_jump

    def get_action(self, player: PlayerData, game_state: GameState, prev_action: np.ndarray) -> List:
        # Reset sequence if it's the first step
        if self.step == 0:
            self.flip_time = game_state.time
            self.step = 1
            
        controls = prev_action.copy()
        
        # Step 1: Backflip (Jump + Pitch Back)
        if self.step == 1:
            controls[5] = 1.0 # Jump
            controls[2] = 1.0 # Pitch back
            
            if game_state.time - self.flip_time >= self.flip_duration:
                self.cancel_time = game_state.time
                self.step = 2
        
        # Step 2: Flip Cancel (Pitch Forward)
        elif self.step == 2:
            controls[5] = 0.0 # Release jump
            controls[2] = -1.0 # Pitch forward (cancel)
            
            if game_state.time - self.cancel_time >= self.cancel_duration:
                self.roll_time = game_state.time
                self.step = 3
                
        # Step 3: Air Roll (Yaw/Roll)
        elif self.step == 3:
            controls[2] = 0.0 # Release pitch
            controls[4] = 1.0 # Roll (or use yaw for a simpler version)
            
            if game_state.time - self.roll_time >= self.roll_duration:
                self.step = 4
                
        # Step 4: Finish
        elif self.step == 4:
            # Release all controls
            controls[4] = 0.0 # Roll
            
            # Sequence is complete, reset for next time
            self.step = 0
            
        return controls.tolist()

    def is_finished(self) -> bool:
        return self.step == 0

    def reset(self):
        self.step = 0
        self.flip_time = 0.0
        self.cancel_time = 0.0
        self.roll_time = 0.0
