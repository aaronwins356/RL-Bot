import numpy as np
from typing import List
from util.game_state import GameState, PlayerData
from sequences.sequence import Sequence

class HalfFlip(Sequence):
    """
    HalfFlip mechanic: Backflip + flip cancel + air roll
    Used for quick 180-degree turns when moving backward or wrong direction.
    Fixed version: Uses state-based logic instead of time-based.
    """
    def __init__(self):
        self.state = 'idle'
        self.initial_height = 0.0
        self.flip_started = False
        self.cancel_applied = False

    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if halfflip can be executed"""
        # Must be in idle state
        if self.state != 'idle':
            return False
        
        # Must be on ground
        on_ground = player.on_ground
        
        # Must have flip available
        has_flip = player.has_flip
        
        # Check if car is moving backward relative to its forward vector
        car = player.car_data
        velocity = car.linear_velocity
        forward = car.forward()
        
        # Dot product: negative means moving backward
        velocity_forward = np.dot(velocity, forward)
        moving_backward = velocity_forward < -200  # Moving backward at >200 uu/s
        
        # Alternative: Check if we want to turn around (moving wrong direction)
        speed = np.linalg.norm(velocity)
        moving_slowly = speed < 600
        
        # Halfflip is valid if moving backward OR moving slowly and want to turn around
        should_halfflip = on_ground and has_flip and (moving_backward or moving_slowly)
        
        return should_halfflip

    def get_action(self, player: PlayerData, game_state: GameState, prev_action: np.ndarray) -> List:
        """Execute halfflip sequence based on physics state"""
        car = player.car_data
        
        if self.state == 'idle':
            # Start sequence - first jump
            self.state = 'first_jump'
            self.initial_height = car.position[2]
            self.flip_started = False
            self.cancel_applied = False
            # Jump
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        elif self.state == 'first_jump':
            # Wait briefly then initiate backflip
            if not player.on_ground or car.position[2] > self.initial_height + 5:
                self.state = 'backflip'
                self.flip_started = True
                # Backflip (pitch back + jump)
                return [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            else:
                # Hold jump briefly
                return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        elif self.state == 'backflip':
            # Check if we're rotating backward (pitch is changing)
            pitch = car.pitch()
            
            # Once we've started flipping, apply cancel
            if not player.on_ground and self.flip_started:
                self.state = 'cancel'
                self.cancel_applied = True
                # Cancel flip (pitch forward hard)
                return [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # Continue backflip
                return [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        elif self.state == 'cancel':
            # Hold cancel to stop backflip rotation
            car_upside_down = abs(car.pitch()) > np.pi * 0.4  # Check if close to inverted
            
            if car_upside_down and not player.on_ground:
                self.state = 'air_roll'
                # Now air roll to land on wheels
                return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
            elif player.on_ground:
                # Landed early - finish
                self.state = 'idle'
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # Continue cancel
                return [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif self.state == 'air_roll':
            # Air roll to land on wheels
            if player.on_ground:
                # Landed - sequence complete
                self.state = 'idle'
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                # Check if car is right-side-up
                up_vector = car.up()
                is_upright = up_vector[2] > 0.8  # Z component of up vector should be positive
                
                if is_upright:
                    # Stop rolling, prepare to land
                    return [1.0, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0, 0.0]
                else:
                    # Continue air roll
                    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        
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
        self.flip_started = False
        self.cancel_applied = False
