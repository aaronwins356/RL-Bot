"""
Flick Mechanic
Executes flicks to shoot the ball with sudden acceleration.
Critical for powerful, unpredictable shots.
"""

import numpy as np
from typing import List, Optional
from util.player_data import PlayerData
from util.game_state import GameState


class FlickMechanic:
    """
    Flick: Jump + delay + dodge to shoot ball
    Provides powerful shots when dribbling or ball is on car.
    """
    
    # States for flick execution
    STATE_IDLE = 'idle'
    STATE_SETUP = 'setup'
    STATE_JUMP = 'jump'
    STATE_DELAY = 'delay'
    STATE_FLICK = 'flick'
    STATE_RECOVER = 'recover'
    
    def __init__(self, flick_direction: Optional[np.ndarray] = None):
        """Initialize flick mechanic.
        
        Args:
            flick_direction: Direction to flick (None = forward)
        """
        self.flick_direction = flick_direction
        self.state = self.STATE_IDLE
        self.jump_time = 0.0
        self.delay_duration = 0.15  # Seconds to wait before dodge
        
    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if flick can be executed.
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            True if flick is valid
        """
        # Must be on ground
        on_ground = player.on_ground
        
        # Must have flip
        has_flip = player.has_flip
        
        # Ball should be close (on car or dribbling)
        ball_pos = game_state.ball.position
        car_pos = player.car_data.position
        ball_distance = np.linalg.norm(ball_pos - car_pos)
        
        # Ball is on car if within ~120 units and above car
        ball_on_car = (
            ball_distance < 150 and
            ball_pos[2] > car_pos[2] + 30  # Ball above car
        )
        
        return on_ground and has_flip and ball_on_car
    
    def execute(
        self,
        player: PlayerData,
        game_state: GameState,
        prev_action: np.ndarray,
        delta_time: float = 1/120.0
    ) -> List[float]:
        """Execute flick sequence.
        
        Args:
            player: Current player
            game_state: Current game state
            prev_action: Previous action
            delta_time: Time since last update (seconds)
            
        Returns:
            Action controls
        """
        if self.state == self.STATE_IDLE:
            # Start flick
            self.state = self.STATE_SETUP
            self.jump_time = 0.0
        
        if self.state == self.STATE_SETUP:
            # Brief setup: ensure we're balanced
            # TODO: Could add balancing logic here
            self.state = self.STATE_JUMP
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif self.state == self.STATE_JUMP:
            # Jump
            self.jump_time = 0.0
            self.state = self.STATE_DELAY
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # Jump
        
        elif self.state == self.STATE_DELAY:
            # Wait before dodge
            self.jump_time += delta_time
            
            if self.jump_time >= self.delay_duration:
                self.state = self.STATE_FLICK
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Release jump
            else:
                # Hold position, slight pitch adjustment
                return [1.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif self.state == self.STATE_FLICK:
            # Execute dodge
            # Determine flick direction
            if self.flick_direction is None:
                # Default: forward flick
                pitch = 1.0
                yaw = 0.0
            else:
                # Custom direction
                pitch = self.flick_direction[0]
                yaw = self.flick_direction[1]
            
            self.state = self.STATE_RECOVER
            return [1.0, 0.0, pitch, yaw, 0.0, 1.0, 0.0, 0.0]  # Dodge
        
        elif self.state == self.STATE_RECOVER:
            # Brief recovery
            self.state = self.STATE_IDLE
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Fallback
        return prev_action.tolist()
    
    def set_direction(self, direction: np.ndarray):
        """Set flick direction.
        
        Args:
            direction: [pitch, yaw] for flick direction
        """
        self.flick_direction = direction
    
    def is_finished(self) -> bool:
        """Check if flick is complete."""
        return self.state == self.STATE_IDLE
    
    def reset(self):
        """Reset flick state."""
        self.state = self.STATE_IDLE
        self.jump_time = 0.0


class MustyFlick(FlickMechanic):
    """
    Musty Flick: Backward flick variation
    Jump + pitch back + delay + front flip
    """
    
    def __init__(self):
        super().__init__(flick_direction=np.array([1.0, 0.0]))
        self.delay_duration = 0.2  # Slightly longer delay
    
    def execute(
        self,
        player: PlayerData,
        game_state: GameState,
        prev_action: np.ndarray,
        delta_time: float = 1/120.0
    ) -> List[float]:
        """Execute musty flick sequence."""
        if self.state == self.STATE_DELAY:
            # Pitch back during delay
            self.jump_time += delta_time
            
            if self.jump_time >= self.delay_duration:
                self.state = self.STATE_FLICK
                return [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Pitch back
            else:
                return [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Hold pitch back
        
        # Use base class for other states
        return super().execute(player, game_state, prev_action, delta_time)


class TurtleFlick(FlickMechanic):
    """
    Turtle Flick: Upside-down flick
    Requires car to be inverted with ball on underside
    """
    
    def __init__(self):
        super().__init__(flick_direction=np.array([-1.0, 0.0]))
        self.delay_duration = 0.1  # Shorter delay
    
    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if turtle flick is valid (car must be inverted)."""
        # Check car orientation (up vector pointing down)
        car_up = player.car_data.up()
        is_inverted = car_up[2] < -0.5  # Mostly upside down
        
        # Ball should be below car
        ball_pos = game_state.ball.position
        car_pos = player.car_data.position
        ball_below = ball_pos[2] < car_pos[2] - 30
        
        # Base checks
        base_valid = super().is_valid(player, game_state)
        
        return base_valid and is_inverted and ball_below
