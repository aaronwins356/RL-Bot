"""
Ball prediction utilities for anticipating future ball positions.
Helps agent plan aerials, intercepts, and positioning.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BallPrediction:
    """
    Predicted ball state at a future time.
    """
    time: float  # Time in seconds
    position: np.ndarray  # 3D position
    velocity: np.ndarray  # 3D velocity
    angular_velocity: np.ndarray  # 3D angular velocity


class SimpleBallPredictor:
    """
    Simple physics-based ball predictor using Rocket League physics.
    Predicts ball trajectory using basic physics simulation.
    """
    
    def __init__(self, gravity: float = -650, drag: float = 0.03, tick_rate: float = 120):
        """
        Args:
            gravity: Gravity constant (Rocket League uses -650)
            drag: Air drag coefficient
            tick_rate: Physics simulation rate (120 Hz in RL)
        """
        self.gravity = gravity
        self.drag = drag
        self.tick_rate = tick_rate
        self.dt = 1.0 / tick_rate
        
        # Field dimensions (Rocket League standard field)
        self.field_length = 10240  # -5120 to 5120
        self.field_width = 8192   # -4096 to 4096
        self.field_height = 2044  # 0 to 2044
        self.ceiling_height = 2044
        self.wall_restitution = 0.6  # Bounce coefficient
    
    def predict(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        num_steps: int = 120,
        step_size: int = 1
    ) -> List[BallPrediction]:
        """
        Predict ball trajectory for future timesteps.
        
        Args:
            ball_pos: Current ball position [x, y, z]
            ball_vel: Current ball velocity [vx, vy, vz]
            ball_ang_vel: Current ball angular velocity
            num_steps: Number of steps to predict (default 1 second at 120Hz)
            step_size: Step size for predictions (1 = every tick, 6 = every 0.05s)
            
        Returns:
            List of BallPrediction objects
        """
        predictions = []
        
        # Copy to avoid modifying originals
        pos = ball_pos.copy()
        vel = ball_vel.copy()
        ang_vel = ball_ang_vel.copy()
        
        for step in range(num_steps):
            # Apply gravity
            vel[2] += self.gravity * self.dt
            
            # Apply drag (simplified)
            speed = np.linalg.norm(vel)
            if speed > 0:
                drag_force = -self.drag * speed
                vel += (vel / speed) * drag_force * self.dt
            
            # Update position
            pos += vel * self.dt
            
            # Check collisions with walls and ground
            pos, vel = self._handle_collisions(pos, vel)
            
            # Store prediction at intervals
            if step % step_size == 0:
                time = step * self.dt
                predictions.append(BallPrediction(
                    time=time,
                    position=pos.copy(),
                    velocity=vel.copy(),
                    angular_velocity=ang_vel.copy()
                ))
        
        return predictions
    
    def _handle_collisions(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle ball collisions with field boundaries.
        Simplified collision model.
        """
        # Ground collision
        ball_radius = 91.25  # Rocket League ball radius
        if pos[2] < ball_radius:
            pos[2] = ball_radius
            vel[2] = -vel[2] * self.wall_restitution
            # Add some horizontal damping
            vel[:2] *= 0.9
        
        # Ceiling collision
        if pos[2] > self.ceiling_height - ball_radius:
            pos[2] = self.ceiling_height - ball_radius
            vel[2] = -vel[2] * self.wall_restitution
        
        # Side walls (Y direction)
        if pos[1] > self.field_length / 2 - ball_radius:
            pos[1] = self.field_length / 2 - ball_radius
            vel[1] = -vel[1] * self.wall_restitution
        elif pos[1] < -self.field_length / 2 + ball_radius:
            pos[1] = -self.field_length / 2 + ball_radius
            vel[1] = -vel[1] * self.wall_restitution
        
        # Side walls (X direction)
        if pos[0] > self.field_width / 2 - ball_radius:
            pos[0] = self.field_width / 2 - ball_radius
            vel[0] = -vel[0] * self.wall_restitution
        elif pos[0] < -self.field_width / 2 + ball_radius:
            pos[0] = -self.field_width / 2 + ball_radius
            vel[0] = -vel[0] * self.wall_restitution
        
        return pos, vel
    
    def get_landing_prediction(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        max_time: float = 3.0
    ) -> Optional[BallPrediction]:
        """
        Find when and where ball will land on ground.
        
        Args:
            ball_pos: Current ball position
            ball_vel: Current ball velocity
            ball_ang_vel: Current ball angular velocity
            max_time: Maximum time to look ahead
            
        Returns:
            BallPrediction when ball lands, or None if doesn't land in time
        """
        num_steps = int(max_time * self.tick_rate)
        predictions = self.predict(ball_pos, ball_vel, ball_ang_vel, num_steps, step_size=6)
        
        ball_radius = 91.25
        ground_threshold = ball_radius + 10  # Slightly above ground
        
        for pred in predictions:
            if pred.position[2] <= ground_threshold and pred.velocity[2] <= 0:
                return pred
        
        return None


class PredictionFeatureExtractor:
    """
    Extracts useful features from ball predictions for agent observations.
    """
    
    def __init__(self, predictor: SimpleBallPredictor):
        self.predictor = predictor
    
    def get_prediction_features(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        player_pos: np.ndarray,
        num_predictions: int = 5
    ) -> np.ndarray:
        """
        Extract prediction features for agent observation.
        
        Args:
            ball_pos: Current ball position
            ball_vel: Current ball velocity
            ball_ang_vel: Current ball angular velocity
            player_pos: Player position
            num_predictions: Number of future positions to include
            
        Returns:
            Feature vector with predicted positions and times
        """
        # Get predictions at intervals (e.g., 0.2s, 0.4s, 0.6s, 0.8s, 1.0s)
        step_size = int(0.2 * self.predictor.tick_rate)  # 24 ticks = 0.2s
        predictions = self.predictor.predict(
            ball_pos, ball_vel, ball_ang_vel,
            num_steps=num_predictions * step_size,
            step_size=step_size
        )
        
        features = []
        
        # Add relative positions of future ball states
        for i in range(min(num_predictions, len(predictions))):
            pred = predictions[i]
            # Relative position to player
            rel_pos = (pred.position - player_pos) / 4096  # Normalized
            features.extend(rel_pos)
        
        # Pad if fewer predictions
        while len(features) < num_predictions * 3:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def get_aerial_opportunity(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        player_pos: np.ndarray,
        player_vel: np.ndarray,
        min_height: float = 300,
        max_time: float = 2.0
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Determine if there's an aerial opportunity.
        
        Args:
            ball_pos: Current ball position
            ball_vel: Current ball velocity
            ball_ang_vel: Current ball angular velocity
            player_pos: Player position
            player_vel: Player velocity
            min_height: Minimum height for aerial
            max_time: Maximum time to reach ball
            
        Returns:
            (has_opportunity, time_to_ball, predicted_ball_pos)
        """
        num_steps = int(max_time * self.predictor.tick_rate)
        predictions = self.predictor.predict(
            ball_pos, ball_vel, ball_ang_vel,
            num_steps=num_steps,
            step_size=6
        )
        
        for pred in predictions:
            # Check if ball is at good aerial height
            if pred.position[2] < min_height:
                continue
            
            # Check if player can reach it (simple distance check)
            dist = np.linalg.norm(pred.position - player_pos)
            time_to_reach = pred.time
            
            # Rough estimate: can player reach in time? (assuming boost)
            max_speed = 2300  # Max car speed with boost
            can_reach = dist / (max_speed * time_to_reach + 1e-6) < 1.2
            
            if can_reach:
                return True, pred.time, pred.position
        
        return False, 0.0, ball_pos
