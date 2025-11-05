"""
Ball Prediction System
Predicts ball trajectory forward in time using physics simulation.
Critical for SSL-level play - enables reading bounces, planning aerials, and timing shots.
"""

import numpy as np
from typing import List, Tuple, Optional
from .physics_object import PhysicsObject


class BallPrediction:
    """Stores a predicted ball state at a future time"""
    def __init__(self, position, velocity, angular_velocity, time):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.angular_velocity = np.array(angular_velocity)
        self.time = time


class BallPredictor:
    """
    Predicts ball trajectory using physics simulation.
    Accounts for gravity, drag, and bounce physics.
    """
    
    # Rocket League physics constants
    GRAVITY = np.array([0, 0, -650])  # uu/s²
    BALL_RADIUS = 91.25  # Unreal Units
    BALL_MASS = 30.0  # kg (approximate)
    DRAG_COEFFICIENT = 0.03  # Air resistance
    RESTITUTION = 0.6  # Bounce elasticity (0 = no bounce, 1 = perfect bounce)
    FRICTION = 0.35  # Surface friction
    
    # Field boundaries (standard arena)
    FIELD_LENGTH = 10240  # x bounds: -5120 to 5120
    FIELD_WIDTH = 8192   # y bounds: -4096 to 4096
    FIELD_HEIGHT = 2044  # z bounds: 0 to 2044 (ceiling)
    
    def __init__(self, prediction_horizon=4.0, timestep=1/60):
        """
        Args:
            prediction_horizon: How many seconds to predict forward (default 4.0)
            timestep: Simulation timestep in seconds (default 1/60)
        """
        self.prediction_horizon = prediction_horizon
        self.timestep = timestep
        self.num_steps = int(prediction_horizon / timestep)
        
    def predict_trajectory(self, ball_state: PhysicsObject) -> List[BallPrediction]:
        """
        Simulate ball physics forward in time.
        
        Args:
            ball_state: Current ball state (position, velocity, angular_velocity)
            
        Returns:
            List of BallPrediction objects, one per timestep
        """
        predictions = []
        
        # Initialize simulation state
        pos = ball_state.position.copy()
        vel = ball_state.linear_velocity.copy()
        ang_vel = ball_state.angular_velocity.copy()
        time = 0.0
        
        for step in range(self.num_steps):
            # Update physics
            pos, vel, ang_vel = self._simulate_step(pos, vel, ang_vel, self.timestep)
            time += self.timestep
            
            # Store prediction
            predictions.append(BallPrediction(pos, vel, ang_vel, time))
        
        return predictions
    
    def _simulate_step(self, position, velocity, angular_velocity, dt):
        """
        Simulate one physics timestep.
        
        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            angular_velocity: Current angular velocity [wx, wy, wz]
            dt: Timestep duration
            
        Returns:
            (new_position, new_velocity, new_angular_velocity)
        """
        # Apply gravity
        acceleration = self.GRAVITY.copy()
        
        # Apply drag (air resistance)
        speed = np.linalg.norm(velocity)
        if speed > 0:
            drag_force = -self.DRAG_COEFFICIENT * speed * velocity
            acceleration += drag_force / self.BALL_MASS
        
        # Update velocity and position
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        
        # Angular velocity decays slightly (spin drag)
        new_angular_velocity = angular_velocity * 0.999
        
        # Check for collisions with field boundaries
        new_position, new_velocity, new_angular_velocity = self._handle_collisions(
            new_position, new_velocity, new_angular_velocity
        )
        
        return new_position, new_velocity, new_angular_velocity
    
    def _handle_collisions(self, position, velocity, angular_velocity):
        """
        Handle collisions with walls, floor, and ceiling.
        Applies bounce physics and friction.
        """
        pos = position.copy()
        vel = velocity.copy()
        ang_vel = angular_velocity.copy()
        
        # Ground collision (z = 0 + ball_radius)
        if pos[2] <= self.BALL_RADIUS:
            pos[2] = self.BALL_RADIUS
            vel[2] = -vel[2] * self.RESTITUTION  # Bounce
            vel[0] *= (1 - self.FRICTION)  # Friction in x
            vel[1] *= (1 - self.FRICTION)  # Friction in y
        
        # Ceiling collision (z = FIELD_HEIGHT - ball_radius)
        if pos[2] >= self.FIELD_HEIGHT - self.BALL_RADIUS:
            pos[2] = self.FIELD_HEIGHT - self.BALL_RADIUS
            vel[2] = -vel[2] * self.RESTITUTION
        
        # Side wall collisions (x = ±FIELD_LENGTH/2 ± ball_radius)
        if pos[0] <= -self.FIELD_LENGTH/2 + self.BALL_RADIUS:
            pos[0] = -self.FIELD_LENGTH/2 + self.BALL_RADIUS
            vel[0] = -vel[0] * self.RESTITUTION
        elif pos[0] >= self.FIELD_LENGTH/2 - self.BALL_RADIUS:
            pos[0] = self.FIELD_LENGTH/2 - self.BALL_RADIUS
            vel[0] = -vel[0] * self.RESTITUTION
        
        # Back wall collisions (y = ±FIELD_WIDTH/2 ± ball_radius)
        if pos[1] <= -self.FIELD_WIDTH/2 + self.BALL_RADIUS:
            pos[1] = -self.FIELD_WIDTH/2 + self.BALL_RADIUS
            vel[1] = -vel[1] * self.RESTITUTION
        elif pos[1] >= self.FIELD_WIDTH/2 - self.BALL_RADIUS:
            pos[1] = self.FIELD_WIDTH/2 - self.BALL_RADIUS
            vel[1] = -vel[1] * self.RESTITUTION
        
        return pos, vel, ang_vel
    
    def get_ball_at_time(self, ball_state: PhysicsObject, target_time: float) -> Optional[BallPrediction]:
        """
        Get predicted ball state at a specific time in the future.
        
        Args:
            ball_state: Current ball state
            target_time: Time in seconds from now
            
        Returns:
            BallPrediction at target_time, or None if beyond prediction horizon
        """
        if target_time > self.prediction_horizon:
            return None
        
        predictions = self.predict_trajectory(ball_state)
        
        # Find closest prediction to target time
        idx = min(int(target_time / self.timestep), len(predictions) - 1)
        return predictions[idx]
    
    def get_landing_time(self, ball_state: PhysicsObject) -> Optional[float]:
        """
        Predict when the ball will next hit the ground.
        
        Args:
            ball_state: Current ball state
            
        Returns:
            Time in seconds until ball hits ground, or None if doesn't land in horizon
        """
        predictions = self.predict_trajectory(ball_state)
        
        # Find first time ball is on ground (z <= ball_radius + small epsilon)
        ground_threshold = self.BALL_RADIUS + 10  # Small epsilon for numerical error
        
        for pred in predictions:
            if pred.position[2] <= ground_threshold:
                return pred.time
        
        return None  # Ball doesn't land within prediction horizon
    
    def get_landing_position(self, ball_state: PhysicsObject) -> Optional[np.ndarray]:
        """
        Predict where the ball will land.
        
        Args:
            ball_state: Current ball state
            
        Returns:
            Landing position [x, y, z], or None if doesn't land in horizon
        """
        predictions = self.predict_trajectory(ball_state)
        
        ground_threshold = self.BALL_RADIUS + 10
        
        for pred in predictions:
            if pred.position[2] <= ground_threshold:
                return pred.position
        
        return None
    
    def get_intercept_time(self, ball_state: PhysicsObject, car_position: np.ndarray, 
                          car_max_speed: float = 2300) -> Optional[Tuple[float, np.ndarray]]:
        """
        Find when and where a car can intercept the ball.
        
        Args:
            ball_state: Current ball state
            car_position: Current car position
            car_max_speed: Maximum car speed (default 2300 uu/s)
            
        Returns:
            (intercept_time, intercept_position) or None if no intercept possible
        """
        predictions = self.predict_trajectory(ball_state)
        
        for pred in predictions:
            # Distance from car to predicted ball position
            distance = np.linalg.norm(pred.position - car_position)
            
            # Time needed for car to reach that position
            time_needed = distance / car_max_speed
            
            # If car can reach in time, this is an intercept point
            if time_needed <= pred.time:
                return (pred.time, pred.position)
        
        return None
    
    def will_ball_bounce_before(self, ball_state: PhysicsObject, target_time: float) -> bool:
        """
        Check if ball will bounce before target_time.
        Useful for deciding between ground shot vs aerial.
        
        Args:
            ball_state: Current ball state
            target_time: Time to check before
            
        Returns:
            True if ball will bounce, False otherwise
        """
        landing_time = self.get_landing_time(ball_state)
        
        if landing_time is None:
            return False
        
        return landing_time < target_time
    
    def is_ball_going_towards_goal(self, ball_state: PhysicsObject, 
                                   goal_position: np.ndarray,
                                   time_horizon: float = 2.0) -> bool:
        """
        Check if ball is heading towards a goal.
        
        Args:
            ball_state: Current ball state
            goal_position: Goal center position [x, y, z]
            time_horizon: How far ahead to check (default 2s)
            
        Returns:
            True if ball is predicted to reach goal area
        """
        # Get ball position at time_horizon
        future_ball = self.get_ball_at_time(ball_state, time_horizon)
        
        if future_ball is None:
            return False
        
        # Check if future position is closer to goal than current position
        current_dist = np.linalg.norm(ball_state.position - goal_position)
        future_dist = np.linalg.norm(future_ball.position - goal_position)
        
        # Also check if ball is actually near goal
        goal_threshold = 1500  # Within 1500 units of goal
        
        return future_dist < current_dist and future_dist < goal_threshold
