"""Fast Aerial Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_FastAerial(SkillProgram):
    """Fast aerial execution with 2-jump pattern.
    
    Trigger: ball height > 400uu and rising OR intercept ETA < 1.2s and ground distance > 900uu
    
    Sequence:
    1. Jump; after 2-4 frames, hold boost + pitch up 0.6-0.9
    2. Second jump within 10-12 frames; yaw alignment toward intercept
    3. Air-roll to match approach plane; feather boost to hit target speed
    
    Termination: car within 120uu of target intercept and contact occurs OR timeout 1.2s
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fast aerial SP."""
        super().__init__(config)
        self.timeout = config.get('timeout', 1.2)
        
        self.inter_jump_frames = config.get('inter_jump_frames', [10, 12])
        self.pitch_up_range = config.get('pitch_up', [0.6, 0.9])
        self.contact_threshold = config.get('contact_threshold', 120.0)
        
        self.first_jump_frame: Optional[int] = None
        self.second_jump_frame: Optional[int] = None
        self.target_intercept: Optional[np.ndarray] = None
        
    def reset(self, obs: Dict[str, Any]) -> None:
        """Reset fast aerial state."""
        self.first_jump_frame = None
        self.second_jump_frame = None
        
        # Compute target intercept point
        ball_pos = obs.get('ball_position', np.zeros(3))
        ball_vel = obs.get('ball_velocity', np.zeros(3))
        
        # Simple prediction: ball position in 0.5s
        self.target_intercept = ball_pos + ball_vel * 0.5
        
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        """Execute fast aerial policy."""
        current_frame = obs.get('frame', 0)
        car_pos = obs.get('car_position', np.zeros(3))
        car_vel = obs.get('car_velocity', np.zeros(3))
        car_orient = obs.get('car_orientation', np.eye(3))
        ball_pos = obs.get('ball_position', np.zeros(3))
        boost = obs.get('boost', 0)
        
        targets = LowLevelTargets()
        
        # Phase 1: First jump
        if self.first_jump_frame is None:
            self.first_jump_frame = current_frame
            targets.jump = True
            targets.boost = False
            targets.pitch = 0.0
            
        # Phase 2: Boost + pitch up (after 2-4 frames)
        elif (self.second_jump_frame is None and 
              current_frame - self.first_jump_frame >= 2 and
              current_frame - self.first_jump_frame < self.inter_jump_frames[0]):
            targets.jump = False
            targets.boost = boost > 10
            
            # Compute pitch to aim at target
            to_target = self.target_intercept - car_pos
            target_dir = to_target / (np.linalg.norm(to_target) + 1e-6)
            
            # Car forward direction
            car_forward = car_orient[:, 0]
            
            # Compute pitch needed
            pitch_error = np.dot(target_dir, car_forward)
            targets.pitch = np.clip(0.7 + pitch_error * 0.3, 
                                   self.pitch_up_range[0], 
                                   self.pitch_up_range[1])
            
        # Phase 3: Second jump
        elif (self.second_jump_frame is None and
              self.inter_jump_frames[0] <= current_frame - self.first_jump_frame <= self.inter_jump_frames[1]):
            self.second_jump_frame = current_frame
            targets.jump = True
            targets.boost = boost > 10
            
            # Yaw alignment toward intercept
            to_target = self.target_intercept - car_pos
            to_target_2d = to_target[:2]  # x, y only
            target_angle = np.arctan2(to_target_2d[1], to_target_2d[0])
            
            car_forward = car_orient[:, 0]
            car_angle = np.arctan2(car_forward[1], car_forward[0])
            
            yaw_error = target_angle - car_angle
            # Normalize to [-pi, pi]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            
            targets.yaw = np.clip(yaw_error * 2.0, -1.0, 1.0)
            targets.pitch = 0.6
            
        # Phase 4: Air control and feather boost
        else:
            targets.jump = False
            
            # Compute direction to target
            to_target = self.target_intercept - car_pos
            distance = np.linalg.norm(to_target)
            target_dir = to_target / (distance + 1e-6)
            
            car_forward = car_orient[:, 0]
            
            # Align car to target
            alignment = np.dot(car_forward, target_dir)
            
            # Compute pitch and yaw
            targets.pitch = np.clip((1.0 - alignment) * target_dir[2], -1.0, 1.0)
            
            to_target_2d = to_target[:2]
            target_angle = np.arctan2(to_target_2d[1], to_target_2d[0])
            car_angle = np.arctan2(car_forward[1], car_forward[0])
            yaw_error = target_angle - car_angle
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            targets.yaw = np.clip(yaw_error, -1.0, 1.0)
            
            # Feather boost based on distance and speed
            current_speed = np.linalg.norm(car_vel)
            desired_speed = min(2300, distance * 3.0)
            
            targets.boost = boost > 10 and current_speed < desired_speed and alignment > 0.7
            
            # Air roll to match approach plane (simplified)
            targets.roll = 0.0
        
        # Check for contact
        distance_to_ball = np.linalg.norm(ball_pos - car_pos)
        success = distance_to_ball < self.contact_threshold
        
        # Compute metrics
        metrics = {
            'distance_to_target': np.linalg.norm(self.target_intercept - car_pos),
            'distance_to_ball': distance_to_ball,
            'boost_used': obs.get('initial_boost', 100) - boost,
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
        
        # Terminate if made contact
        distance = np.linalg.norm(ball_pos - car_pos)
        if distance < self.contact_threshold:
            return True
        
        # Terminate if target intercept reached
        if self.target_intercept is not None:
            dist_to_target = np.linalg.norm(self.target_intercept - car_pos)
            if dist_to_target < self.contact_threshold:
                return True
        
        return False
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        """Get fallback SP."""
        # Fall back to aerial control or recovery
        return 'SP_AerialControl'
