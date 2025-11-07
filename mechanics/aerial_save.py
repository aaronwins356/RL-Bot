"""
Aerial Save Mechanic
Executes aerial saves to prevent goals.
Critical for defensive positioning and shot blocking.
"""

import numpy as np
from typing import List, Optional, Tuple
from util.player_data import PlayerData
from util.game_state import GameState


class AerialSaveMechanic:
    """
    Aerial Save: Fast aerial to intercept ball heading toward goal.
    Prioritizes timing and positioning to block shots.
    """
    
    STATE_IDLE = 'idle'
    STATE_POSITION = 'position'
    STATE_TAKEOFF = 'takeoff'
    STATE_AERIAL = 'aerial'
    STATE_CONTACT = 'contact'
    STATE_RECOVER = 'recover'
    
    def __init__(self):
        """Initialize aerial save mechanic."""
        self.state = self.STATE_IDLE
        self.intercept_point = None
        self.intercept_time = 0.0
        self.takeoff_initiated = False
        
    def predict_ball_trajectory(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        timesteps: int = 120
    ) -> List[np.ndarray]:
        """Predict ball trajectory with simple physics.
        
        Args:
            ball_pos: Current ball position
            ball_vel: Current ball velocity
            timesteps: Number of future timesteps to predict
            
        Returns:
            List of predicted positions
        """
        GRAVITY = np.array([0, 0, -650])  # uu/s^2
        DRAG = 0.03  # Simple drag coefficient
        dt = 1/120.0  # 120 Hz tick rate
        
        trajectory = []
        pos = ball_pos.copy()
        vel = ball_vel.copy()
        
        for _ in range(timesteps):
            # Apply physics
            vel += GRAVITY * dt
            vel *= (1 - DRAG * dt)  # Simple drag
            pos += vel * dt
            
            # Floor bounce (simplified)
            if pos[2] < 93:  # Ball radius ~93 units
                pos[2] = 93
                vel[2] = abs(vel[2]) * 0.6  # Bounce with damping
            
            trajectory.append(pos.copy())
        
        return trajectory
    
    def calculate_intercept_point(
        self,
        player: PlayerData,
        game_state: GameState
    ) -> Tuple[Optional[np.ndarray], float]:
        """Calculate best intercept point for save.
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            Tuple of (intercept_point, intercept_time) or (None, 0) if no save needed
        """
        ball_pos = game_state.ball.position
        ball_vel = game_state.ball.linear_velocity
        
        # Get own goal position
        goal_pos = game_state.get_goal_position(player.team)
        
        # Predict ball trajectory
        trajectory = self.predict_ball_trajectory(ball_pos, ball_vel, timesteps=240)  # 2 seconds
        
        # Find if ball will enter goal
        GOAL_WIDTH = 1786  # Goal width in units
        GOAL_HEIGHT = 642  # Goal height in units
        
        best_intercept = None
        best_time = 0.0
        
        for i, future_pos in enumerate(trajectory):
            # Check if ball will be in goal area
            x_in_goal = abs(future_pos[0] - goal_pos[0]) < GOAL_WIDTH / 2
            z_in_goal = future_pos[2] < GOAL_HEIGHT
            y_near_goal = abs(future_pos[1] - goal_pos[1]) < 500  # Close to goal line
            
            if x_in_goal and z_in_goal and y_near_goal:
                # Ball heading toward goal, intercept before goal
                # Find best point between ball and goal
                intercept_time = i / 120.0  # Convert to seconds
                
                # Intercept slightly in front of goal
                goal_to_ball = future_pos - goal_pos
                goal_to_ball_norm = goal_to_ball / (np.linalg.norm(goal_to_ball) + 1e-8)
                intercept_distance = min(500, np.linalg.norm(goal_to_ball) * 0.7)
                intercept_point = goal_pos + goal_to_ball_norm * intercept_distance
                
                best_intercept = intercept_point
                best_time = intercept_time
                break  # Take first intercept opportunity
        
        return best_intercept, best_time
    
    def is_valid(self, player: PlayerData, game_state: GameState) -> bool:
        """Check if aerial save is needed and feasible.
        
        Args:
            player: Current player
            game_state: Current game state
            
        Returns:
            True if save should be attempted
        """
        # Must have boost
        has_boost = player.boost_amount > 20.0
        
        # Must have flip or be able to aerial
        can_aerial = player.has_flip or not player.on_ground
        
        # Calculate if ball heading toward goal
        intercept_point, intercept_time = self.calculate_intercept_point(player, game_state)
        save_needed = intercept_point is not None
        
        if not save_needed:
            return False
        
        # Check if we can reach intercept in time
        player_pos = player.car_data.position
        distance_to_intercept = np.linalg.norm(intercept_point - player_pos)
        
        # Assume average aerial speed of 1500 uu/s
        time_to_reach = distance_to_intercept / 1500.0
        can_reach = time_to_reach < intercept_time
        
        return has_boost and can_aerial and save_needed and can_reach
    
    def execute(
        self,
        player: PlayerData,
        game_state: GameState,
        prev_action: np.ndarray
    ) -> List[float]:
        """Execute aerial save sequence.
        
        Args:
            player: Current player
            game_state: Current game state
            prev_action: Previous action
            
        Returns:
            Action controls
        """
        if self.state == self.STATE_IDLE:
            # Calculate intercept
            self.intercept_point, self.intercept_time = self.calculate_intercept_point(
                player, game_state
            )
            
            if self.intercept_point is None:
                return prev_action.tolist()
            
            self.state = self.STATE_POSITION
            self.takeoff_initiated = False
        
        if self.state == self.STATE_POSITION:
            # Quick positioning (mostly skip for urgency)
            self.state = self.STATE_TAKEOFF
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif self.state == self.STATE_TAKEOFF:
            # Fast aerial takeoff
            if not self.takeoff_initiated:
                self.takeoff_initiated = True
                # Jump + pitch + boost
                return [1.0, 0.0, -0.8, 0.0, 0.0, 1.0, 1.0, 0.0]
            else:
                # Transition to aerial
                self.state = self.STATE_AERIAL
                return [1.0, 0.0, -0.8, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        elif self.state == self.STATE_AERIAL:
            # Point toward intercept and boost
            controls = self._calculate_aerial_controls(
                player.car_data,
                self.intercept_point
            )
            
            # Check if close to contact
            car_pos = player.car_data.position
            distance = np.linalg.norm(car_pos - self.intercept_point)
            
            if distance < 200:
                self.state = self.STATE_CONTACT
            
            return controls
        
        elif self.state == self.STATE_CONTACT:
            # Make contact, try to clear
            # Point toward ball and hit it away from goal
            ball_pos = game_state.ball.position
            goal_pos = game_state.get_goal_position(player.team)
            
            # Clear direction: away from goal
            goal_to_ball = ball_pos - goal_pos
            clear_direction = goal_to_ball / (np.linalg.norm(goal_to_ball) + 1e-8)
            
            # Calculate desired position (beyond ball)
            target_pos = ball_pos + clear_direction * 200
            
            controls = self._calculate_aerial_controls(
                player.car_data,
                target_pos
            )
            
            self.state = self.STATE_RECOVER
            return controls
        
        elif self.state == self.STATE_RECOVER:
            # Brief recovery
            self.state = self.STATE_IDLE
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return prev_action.tolist()
    
    def _calculate_aerial_controls(
        self,
        car,
        target: np.ndarray
    ) -> List[float]:
        """Calculate aerial controls toward target.
        
        Args:
            car: Car data
            target: Target position
            
        Returns:
            Controls list
        """
        # Direction to target
        to_target = target - car.position
        distance = np.linalg.norm(to_target)
        
        if distance < 1:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        to_target_norm = to_target / distance
        
        # Current forward direction
        forward = car.forward()
        
        # Pitch control
        pitch = -to_target_norm[2] * 1.5
        pitch = np.clip(pitch, -1, 1)
        
        # Yaw control
        yaw_angle = np.arctan2(
            to_target_norm[1] - forward[1],
            to_target_norm[0] - forward[0]
        )
        yaw = yaw_angle * 0.5
        yaw = np.clip(yaw, -1, 1)
        
        # Boost if aligned
        alignment = np.dot(forward, to_target_norm)
        boost = 1.0 if alignment > 0.7 else 0.0
        
        return [1.0, 0.0, pitch, yaw, 0.0, 0.0, boost, 0.0]
    
    def is_finished(self) -> bool:
        """Check if save is complete."""
        return self.state == self.STATE_IDLE
    
    def reset(self):
        """Reset save state."""
        self.state = self.STATE_IDLE
        self.intercept_point = None
        self.intercept_time = 0.0
        self.takeoff_initiated = False
