"""Low-Level Controller (LLC) for converting high-level targets to control surfaces."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PIDState:
    """PID controller state."""
    
    integral: np.ndarray = None
    prev_error: np.ndarray = None
    
    def __post_init__(self):
        if self.integral is None:
            self.integral = np.zeros(3)
        if self.prev_error is None:
            self.prev_error = np.zeros(3)


class PIDController:
    """PID controller for orientation and position control."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.1, 
                 max_integral: float = 1.0, max_output: float = 1.0):
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_integral: Maximum integral accumulation (anti-windup)
            max_output: Maximum output value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.state = PIDState()
        
    def reset(self):
        """Reset controller state."""
        self.state = PIDState()
        
    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Update PID controller.
        
        Args:
            error: Current error vector
            dt: Time delta
            
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.state.integral += error * dt
        self.state.integral = np.clip(self.state.integral, 
                                     -self.max_integral, self.max_integral)
        i_term = self.ki * self.state.integral
        
        # Derivative term
        if dt > 0:
            d_term = self.kd * (error - self.state.prev_error) / dt
        else:
            d_term = np.zeros_like(error)
        
        self.state.prev_error = error.copy()
        
        # Compute output
        output = p_term + i_term + d_term
        output = np.clip(output, -self.max_output, self.max_output)
        
        return output


class FastAerialHelper:
    """Helper for fast aerial timing and execution."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fast aerial helper.
        
        Args:
            config: Configuration dict
        """
        self.inter_jump_frames_min = config.get('inter_jump_frames', [10, 12])[0]
        self.inter_jump_frames_max = config.get('inter_jump_frames', [10, 12])[1]
        self.pitch_up_min = config.get('pitch_up', [0.6, 0.9])[0]
        self.pitch_up_max = config.get('pitch_up', [0.6, 0.9])[1]
        
        self.first_jump_frame: Optional[int] = None
        self.frame_count = 0
        
    def reset(self):
        """Reset fast aerial state."""
        self.first_jump_frame = None
        self.frame_count = 0
        
    def should_second_jump(self, current_frame: int) -> bool:
        """Check if should execute second jump.
        
        Args:
            current_frame: Current frame number
            
        Returns:
            True if should execute second jump
        """
        if self.first_jump_frame is None:
            return False
        
        frames_since_first = current_frame - self.first_jump_frame
        return (self.inter_jump_frames_min <= frames_since_first 
                <= self.inter_jump_frames_max)
    
    def execute_first_jump(self, current_frame: int) -> Dict[str, Any]:
        """Execute first jump of fast aerial.
        
        Args:
            current_frame: Current frame number
            
        Returns:
            Control dict for first jump
        """
        self.first_jump_frame = current_frame
        return {
            'jump': True,
            'boost': False,
            'pitch': 0.0,
        }
    
    def execute_boost_phase(self, target_pitch: float) -> Dict[str, Any]:
        """Execute boost phase between jumps.
        
        Args:
            target_pitch: Target pitch value
            
        Returns:
            Control dict for boost phase
        """
        pitch = np.clip(target_pitch, self.pitch_up_min, self.pitch_up_max)
        return {
            'jump': False,
            'boost': True,
            'pitch': pitch,
        }
    
    def execute_second_jump(self, yaw_alignment: float) -> Dict[str, Any]:
        """Execute second jump of fast aerial.
        
        Args:
            yaw_alignment: Yaw alignment value
            
        Returns:
            Control dict for second jump
        """
        return {
            'jump': True,
            'boost': True,
            'pitch': 0.7,
            'yaw': yaw_alignment,
        }


class FlipResetDetector:
    """Detector for flip reset opportunities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize flip reset detector.
        
        Args:
            config: Configuration dict
        """
        self.contact_window_ms = config.get('contact_window_ms', 20)
        self.max_relative_velocity = config.get('max_relative_velocity', 150.0)
        
    def detect(self, obs: Dict[str, Any]) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect if flip reset is possible.
        
        Args:
            obs: Current observation
            
        Returns:
            Tuple of (is_possible, contact_point)
        """
        car_pos = obs.get('car_position', np.zeros(3))
        car_vel = obs.get('car_velocity', np.zeros(3))
        car_orient = obs.get('car_orientation', np.eye(3))
        
        ball_pos = obs.get('ball_position', np.zeros(3))
        ball_vel = obs.get('ball_velocity', np.zeros(3))
        ball_radius = obs.get('ball_radius', 91.25)
        
        # Check if car is near ball
        car_to_ball = ball_pos - car_pos
        distance = np.linalg.norm(car_to_ball)
        
        # Need to be within wheel contact range
        if distance > ball_radius + 50:  # 50uu margin for wheels
            return False, None
        
        # Check relative velocity is low
        relative_vel = np.linalg.norm(ball_vel - car_vel)
        if relative_vel > self.max_relative_velocity:
            return False, None
        
        # Check if car bottom (wheels) are toward ball
        car_up = car_orient[:, 2]  # Z-axis of car
        to_ball_normalized = car_to_ball / (distance + 1e-6)
        
        # Car bottom should face ball (up vector points away from ball)
        alignment = -np.dot(car_up, to_ball_normalized)
        
        if alignment < 0.8:  # Need good alignment
            return False, None
        
        # Compute contact point
        contact_point = ball_pos - to_ball_normalized * ball_radius
        
        return True, contact_point
    
    def is_four_wheel_contact(self, obs: Dict[str, Any]) -> bool:
        """Check if all four wheels are in contact with ball.
        
        Args:
            obs: Current observation
            
        Returns:
            True if four-wheel contact detected
        """
        # This requires detailed wheel position tracking
        # For now, use simplified check based on alignment and distance
        is_possible, _ = self.detect(obs)
        
        if not is_possible:
            return False
        
        # Check car is relatively flat against ball
        car_orient = obs.get('car_orientation', np.eye(3))
        car_pos = obs.get('car_position', np.zeros(3))
        ball_pos = obs.get('ball_position', np.zeros(3))
        
        car_to_ball = ball_pos - car_pos
        distance = np.linalg.norm(car_to_ball)
        to_ball_normalized = car_to_ball / (distance + 1e-6)
        
        car_up = car_orient[:, 2]
        alignment = abs(np.dot(car_up, to_ball_normalized))
        
        # Very tight alignment required for four-wheel contact
        return alignment > 0.95


class AirRollStabilizer:
    """Stabilizer for air roll control with angular velocity damping."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize stabilizer.
        
        Args:
            config: Configuration dict
        """
        self.damping_factor = config.get('damping_factor', 0.1)
        self.max_angular_vel = config.get('max_angular_vel', 5.5)  # rad/s
        
    def stabilize(self, current_angular_vel: np.ndarray, 
                  target_angular_vel: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute stabilization controls.
        
        Args:
            current_angular_vel: Current angular velocity [wx, wy, wz]
            target_angular_vel: Target angular velocity (None for damping only)
            
        Returns:
            Control vector [yaw, pitch, roll]
        """
        if target_angular_vel is None:
            target_angular_vel = np.zeros(3)
        
        # Compute error
        error = target_angular_vel - current_angular_vel
        
        # Apply damping
        control = self.damping_factor * error
        
        # Clip to valid range
        control = np.clip(control, -1.0, 1.0)
        
        return control


class LowLevelController:
    """Low-Level Controller (LLC) that converts SP targets to control surfaces."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLC.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Initialize sub-controllers
        self.pid_orientation = PIDController(
            kp=config.get('pid_kp', 1.5),
            ki=config.get('pid_ki', 0.0),
            kd=config.get('pid_kd', 0.3),
        )
        
        self.pid_position = PIDController(
            kp=config.get('pid_pos_kp', 0.5),
            ki=config.get('pid_pos_ki', 0.0),
            kd=config.get('pid_pos_kd', 0.1),
        )
        
        self.fast_aerial = FastAerialHelper(config.get('fast_aerial', {}))
        self.flip_reset_detector = FlipResetDetector(config.get('flip_reset', {}))
        self.air_roll_stabilizer = AirRollStabilizer(config.get('air_roll', {}))
        
        self.dt = config.get('dt', 1.0 / 120.0)  # 120 Hz
        
    def reset(self):
        """Reset all controller states."""
        self.pid_orientation.reset()
        self.pid_position.reset()
        self.fast_aerial.reset()
        
    def compute_controls(self, targets: 'LowLevelTargets', obs: Dict[str, Any],
                        current_frame: int) -> Dict[str, Any]:
        """Compute low-level controls from targets.
        
        Args:
            targets: Target state from skill program
            obs: Current observation
            current_frame: Current frame number
            
        Returns:
            Dict of control values
        """
        controls = {
            'steer': targets.steer,
            'throttle': targets.throttle,
            'yaw': targets.yaw,
            'pitch': targets.pitch,
            'roll': targets.roll,
            'jump': targets.jump,
            'boost': targets.boost,
            'handbrake': targets.handbrake,
        }
        
        # Apply PID for target pose if specified
        if targets.target_orientation is not None:
            # Compute orientation error and apply PID
            # This is simplified; real implementation would use quaternion math
            car_orient = obs.get('car_orientation', np.eye(3))
            # orientation_error = compute_orientation_error(car_orient, targets.target_orientation)
            # orientation_control = self.pid_orientation.update(orientation_error, self.dt)
            # controls['yaw'] = orientation_control[0]
            # controls['pitch'] = orientation_control[1]
            # controls['roll'] = orientation_control[2]
            pass
        
        # Apply stabilization if needed
        if obs.get('wheels_on_ground', 0) == 0:  # In air
            angular_vel = obs.get('angular_velocity', np.zeros(3))
            stab_control = self.air_roll_stabilizer.stabilize(
                angular_vel, targets.target_angular_velocity
            )
            # Blend with existing controls
            controls['yaw'] = np.clip(controls['yaw'] + stab_control[0] * 0.3, -1, 1)
            controls['pitch'] = np.clip(controls['pitch'] + stab_control[1] * 0.3, -1, 1)
            controls['roll'] = np.clip(controls['roll'] + stab_control[2] * 0.3, -1, 1)
        
        return controls
