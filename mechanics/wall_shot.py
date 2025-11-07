"""
Wall Shot Mechanic
Executes shots and aerials from walls.
Critical for utilizing full 3D space and unpredictable angles.
"""

import numpy as np
from typing import List, Optional, Tuple
from util.player_data import PlayerData
from util.game_state import GameState


class WallShotMechanic:
    """
    Wall Shot: Drive on wall + jump + aerial to hit ball
    Provides unique angles and unpredictable shots.
    """
    
    STATE_IDLE = 'idle'
    STATE_APPROACH_WALL = 'approach'
    STATE_DRIVE_WALL = 'drive_wall'
    STATE_TAKEOFF = 'takeoff'
    STATE_AERIAL = 'aerial'
    STATE_RECOVER = 'recover'
    
    # Field dimensions
    FIELD_LENGTH = 10240
    FIELD_WIDTH = 8192
    WALL_HEIGHT = 2044
    
    def __init__(self, target_position: Optional[np.ndarray] = None):
        """Initialize wall shot mechanic.
        
        Args:
            target_position: Target to shoot at (None = opponent goal)
        """
        self.target_position = target_position
        self.state = self.STATE_IDLE
        self.wall_contact_point = None
        self.takeoff_initiated = False
        
    def find_nearest_wall(
        self,
        position: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        """Find nearest wall and contact point.
        
        Args:
            position: Current position
            
        Returns:
            Tuple of (wall_name, wall_normal)
        """
        x, y, z = position
        
        # Distances to each wall
        dist_left = abs(x + self.FIELD_WIDTH / 2)
        dist_right = abs(x - self.FIELD_WIDTH / 2)
        dist_back = abs(y + self.FIELD_LENGTH / 2)
        dist_front = abs(y - self.FIELD_LENGTH / 2)
        
        # Find closest
        min_dist = min(dist_left, dist_right, dist_back, dist_front)
        
        if min_dist == dist_left:
            return 'left', np.array([-1, 0, 0])
        elif min_dist == dist_right:
            return 'right', np.array([1, 0, 0])
        elif min_dist == dist_back:
            return 'back', np.array([0, -1, 0])
        else:
            return 'front', np.array([0, 1, 0])
    
    def is_on_wall(self, player: PlayerData) -> bool:
        """Check if car is on wall.
        
        Args:
            player: Current player
            
        Returns:
            True if on wall
        """
        car_pos = player.car_data.position
        
        # Check if near a wall
        near_left = abs(car_pos[0] + self.FIELD_WIDTH / 2) < 200
        near_right = abs(car_pos[0] - self.FIELD_WIDTH / 2) < 200
        near_back = abs(car_pos[1] + self.FIELD_LENGTH / 2) < 200
        near_front = abs(car_pos[1] - self.FIELD_LENGTH / 2) < 200
        
        near_wall = near_left or near_right or near_back or near_front
        
        # Check if car is oriented toward wall (up vector pointing away from center)
        car_up = player.car_data.up()
        wall_name, wall_normal = self.find_nearest_wall(car_pos)
        
        # Up vector should align with wall normal
        alignment = np.dot(car_up, wall_normal)
        aligned_with_wall = abs(alignment) > 0.7
        
        # Must have wall contact
        has_wall_contact = player.has_wheel_contact
        
        return near_wall and aligned_with_wall and has_wall_contact
    
    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if wall shot can be executed.
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            True if wall shot is feasible
        """
        # Ball should be near a wall
        ball_pos = game_state.ball.position
        ball_near_wall = (
            abs(ball_pos[0]) > self.FIELD_WIDTH / 2 - 1000 or
            abs(ball_pos[1]) > self.FIELD_LENGTH / 2 - 1000
        )
        
        # Ball should be elevated (not on ground)
        ball_elevated = ball_pos[2] > 300
        
        # Player should be near wall or able to reach it
        player_pos = player.car_data.position
        player_near_wall = (
            abs(player_pos[0]) > self.FIELD_WIDTH / 2 - 1500 or
            abs(player_pos[1]) > self.FIELD_LENGTH / 2 - 1500
        )
        
        # Must have boost
        has_boost = player.boost_amount > 15.0
        
        return ball_near_wall and ball_elevated and player_near_wall and has_boost
    
    def execute(
        self,
        player: PlayerData,
        game_state: GameState,
        prev_action: np.ndarray
    ) -> List[float]:
        """Execute wall shot sequence.
        
        Args:
            player: Current player
            game_state: Current game state
            prev_action: Previous action
            
        Returns:
            Action controls
        """
        # Set target if not set
        if self.target_position is None:
            # Default to opponent goal
            opponent_team = 1 - player.team
            self.target_position = game_state.get_goal_position(opponent_team)
        
        if self.state == self.STATE_IDLE:
            # Start sequence
            self.state = self.STATE_APPROACH_WALL
            self.takeoff_initiated = False
        
        if self.state == self.STATE_APPROACH_WALL:
            # Drive toward wall
            if self.is_on_wall(player):
                self.state = self.STATE_DRIVE_WALL
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # Drive toward nearest wall
                wall_name, wall_normal = self.find_nearest_wall(player.car_data.position)
                
                # Simple approach: drive toward wall
                # This is simplified - real implementation would be more sophisticated
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]  # Throttle + some boost
        
        elif self.state == self.STATE_DRIVE_WALL:
            # Drive on wall toward ball
            ball_pos = game_state.ball.position
            car_pos = player.car_data.position
            
            # Calculate if we should take off
            dist_to_ball = np.linalg.norm(ball_pos - car_pos)
            ball_above = ball_pos[2] > car_pos[2] + 100
            
            if dist_to_ball < 500 and ball_above:
                self.state = self.STATE_TAKEOFF
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # Continue driving on wall
                # Simplified - should calculate proper steering
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
        
        elif self.state == self.STATE_TAKEOFF:
            # Jump off wall
            if not self.takeoff_initiated:
                self.takeoff_initiated = True
                # Jump + boost
                return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            else:
                self.state = self.STATE_AERIAL
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        elif self.state == self.STATE_AERIAL:
            # Aerial toward ball and shoot toward target
            ball_pos = game_state.ball.position
            car_pos = player.car_data.position
            
            # Calculate shot direction (through ball toward target)
            ball_to_target = self.target_position - ball_pos
            ball_to_target_norm = ball_to_target / (np.linalg.norm(ball_to_target) + 1e-8)
            
            # Aim point: slightly before ball in direction of target
            aim_point = ball_pos - ball_to_target_norm * 100
            
            controls = self._calculate_aerial_controls(
                player.car_data,
                aim_point
            )
            
            # Check if close to ball
            if np.linalg.norm(car_pos - ball_pos) < 200:
                self.state = self.STATE_RECOVER
            
            return controls
        
        elif self.state == self.STATE_RECOVER:
            # Recovery
            self.state = self.STATE_IDLE
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return prev_action.tolist()
    
    def _calculate_aerial_controls(
        self,
        car,
        target: np.ndarray
    ) -> List[float]:
        """Calculate aerial controls.
        
        Args:
            car: Car data
            target: Target position
            
        Returns:
            Controls list
        """
        to_target = target - car.position
        distance = np.linalg.norm(to_target)
        
        if distance < 1:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        to_target_norm = to_target / distance
        forward = car.forward()
        
        # Pitch
        pitch = -to_target_norm[2] * 1.5
        pitch = np.clip(pitch, -1, 1)
        
        # Yaw
        yaw_angle = np.arctan2(
            to_target_norm[1] - forward[1],
            to_target_norm[0] - forward[0]
        )
        yaw = yaw_angle * 0.5
        yaw = np.clip(yaw, -1, 1)
        
        # Roll (for wall shots, might need roll adjustment)
        # Simplified - not implementing complex roll logic here
        roll = 0.0
        
        # Boost
        alignment = np.dot(forward, to_target_norm)
        boost = 1.0 if alignment > 0.7 else 0.0
        
        return [1.0, 0.0, pitch, yaw, roll, 0.0, boost, 0.0]
    
    def is_finished(self) -> bool:
        """Check if wall shot is complete."""
        return self.state == self.STATE_IDLE
    
    def reset(self):
        """Reset wall shot state."""
        self.state = self.STATE_IDLE
        self.wall_contact_point = None
        self.target_position = None
        self.takeoff_initiated = False
