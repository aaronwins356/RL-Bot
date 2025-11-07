"""Aerial Control Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_AerialControl(SkillProgram):
    """Aerial control with orientation and angular velocity damping.
    
    Goal: Orient body normal to target shot vector; damp angular velocity.
    Control: LLC holds target quaternion; anti-windup PID; optional MPPI for snap turns > 90°
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize aerial control SP."""
        super().__init__(config)
        self.timeout = config.get('timeout', 3.0)
        self.damping_factor = config.get('damping_factor', 0.2)
        self.target_angular_vel_threshold = config.get('target_angular_vel_threshold', 0.5)
        
    def reset(self, obs: Dict[str, Any]) -> None:
        """Reset aerial control state."""
        pass
    
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        """Execute aerial control policy."""
        car_pos = obs.get('car_position', np.zeros(3))
        car_vel = obs.get('car_velocity', np.zeros(3))
        car_orient = obs.get('car_orientation', np.eye(3))
        car_angular_vel = obs.get('angular_velocity', np.zeros(3))
        ball_pos = obs.get('ball_position', np.zeros(3))
        boost = obs.get('boost', 0)
        
        targets = LowLevelTargets()
        
        # Compute target direction (toward ball)
        to_ball = ball_pos - car_pos
        distance = np.linalg.norm(to_ball)
        target_dir = to_ball / (distance + 1e-6)
        
        # Current car forward direction
        car_forward = car_orient[:, 0]
        
        # Compute alignment error
        alignment = np.dot(car_forward, target_dir)
        
        # Compute pitch and yaw to align
        # Pitch: controls up/down
        pitch_component = target_dir[2]  # z component
        targets.pitch = np.clip(pitch_component * 2.0, -1.0, 1.0)
        
        # Yaw: controls left/right
        to_ball_2d = to_ball[:2]
        target_angle = np.arctan2(to_ball_2d[1], to_ball_2d[0])
        car_angle = np.arctan2(car_forward[1], car_forward[0])
        
        yaw_error = target_angle - car_angle
        # Normalize to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        targets.yaw = np.clip(yaw_error * 1.5, -1.0, 1.0)
        
        # Roll: keep level or match approach plane
        targets.roll = 0.0
        
        # Damp angular velocity
        angular_vel_magnitude = np.linalg.norm(car_angular_vel)
        if angular_vel_magnitude > self.target_angular_vel_threshold:
            # Apply damping
            damping = -car_angular_vel * self.damping_factor
            targets.yaw += damping[2] * 0.3  # Yaw damping
            targets.pitch += damping[1] * 0.3  # Pitch damping
            targets.roll += damping[0] * 0.3  # Roll damping
            
            # Clip
            targets.yaw = np.clip(targets.yaw, -1.0, 1.0)
            targets.pitch = np.clip(targets.pitch, -1.0, 1.0)
            targets.roll = np.clip(targets.roll, -1.0, 1.0)
        
        # Boost management: only boost if aligned and moving toward target
        vel_toward_ball = np.dot(car_vel, target_dir)
        should_boost = (alignment > 0.8 and 
                       vel_toward_ball < 1500 and 
                       boost > 10 and 
                       distance > 300)
        
        targets.boost = should_boost
        targets.throttle = 1.0 if not should_boost else 0.0
        
        # Compute success (good alignment and low angular velocity)
        success = alignment > 0.95 and angular_vel_magnitude < self.target_angular_vel_threshold
        
        metrics = {
            'alignment': alignment,
            'angular_vel_magnitude': angular_vel_magnitude,
            'distance_to_ball': distance,
        }
        
        return SkillProgramResult(
            targets=targets,
            success=success,
            metrics=metrics,
        )
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        """Check if should terminate."""
        car_pos = obs.get('car_position', np.zeros(3))
        ball_pos = obs.get('ball_position', np.zeros(3))
        car_orient = obs.get('car_orientation', np.eye(3))
        car_angular_vel = obs.get('angular_velocity', np.zeros(3))
        
        # Terminate if well aligned with low angular velocity
        to_ball = ball_pos - car_pos
        distance = np.linalg.norm(to_ball)
        target_dir = to_ball / (distance + 1e-6)
        car_forward = car_orient[:, 0]
        alignment = np.dot(car_forward, target_dir)
        
        angular_vel_magnitude = np.linalg.norm(car_angular_vel)
        
        if alignment > 0.95 and angular_vel_magnitude < self.target_angular_vel_threshold:
            return True
        
        # Terminate if on ground
        if obs.get('wheels_on_ground', 0) > 0:
            return True
        
        return False
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        """Get fallback SP."""
        # No fallback, this is a basic control SP
        return None
