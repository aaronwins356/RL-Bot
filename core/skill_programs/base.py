"""Base classes for Skill Programs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class LowLevelTargets:
    """Target state for low-level controller."""
    
    # Continuous controls
    steer: float = 0.0  # [-1, 1]
    throttle: float = 0.0  # [-1, 1]
    yaw: float = 0.0  # [-1, 1]
    pitch: float = 0.0  # [-1, 1]
    roll: float = 0.0  # [-1, 1]
    
    # Binary controls
    jump: bool = False
    boost: bool = False
    handbrake: bool = False
    
    # Discrete tech triggers (optional, gated)
    trigger_fast_aerial: bool = False
    trigger_ceiling_setup: bool = False
    trigger_flip_reset: bool = False
    trigger_musty: bool = False
    trigger_breezi: bool = False
    trigger_double_tap: bool = False
    
    # Optional target pose for PID controller
    target_position: Optional[np.ndarray] = None  # [x, y, z]
    target_orientation: Optional[np.ndarray] = None  # quaternion [w, x, y, z]
    target_velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    target_angular_velocity: Optional[np.ndarray] = None  # [wx, wy, wz]


@dataclass
class SkillProgramResult:
    """Result from executing a skill program."""
    
    targets: LowLevelTargets
    should_terminate: bool = False
    timeout_reached: bool = False
    success: bool = False
    fallback_sp: Optional[str] = None  # Name of fallback SP if needed
    metrics: Dict[str, float] = None  # Optional metrics for logging
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class SkillProgram(ABC):
    """Base class for all skill programs.
    
    Each skill program implements a specific mechanic or maneuver with:
    - Trigger conditions (when to activate)
    - Policy execution (how to execute the maneuver)
    - Termination predicates (when to end)
    - Fallback sequences (what to do on failure)
    - Timeout handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize skill program.
        
        Args:
            config: Configuration dict for this SP
        """
        self.config = config
        self.name = self.__class__.__name__
        self.timeout = config.get('timeout', 2.0)  # Default 2.0s timeout
        self.start_time: Optional[float] = None
        self.active = False
        
    @abstractmethod
    def reset(self, obs: Dict[str, Any]) -> None:
        """Reset skill program state.
        
        Args:
            obs: Current observation dictionary
        """
        pass
    
    @abstractmethod
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        """Execute policy for this skill program.
        
        Args:
            obs: Current observation dictionary
            
        Returns:
            SkillProgramResult with targets and status
        """
        pass
    
    @abstractmethod
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        """Check if this skill program should terminate.
        
        Args:
            obs: Current observation dictionary
            
        Returns:
            True if should terminate
        """
        pass
    
    @abstractmethod
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        """Get fallback skill program on failure.
        
        Args:
            obs: Current observation dictionary
            
        Returns:
            Name of fallback SP, or None
        """
        pass
    
    def activate(self, obs: Dict[str, Any], current_time: float) -> None:
        """Activate this skill program.
        
        Args:
            obs: Current observation
            current_time: Current simulation time
        """
        self.active = True
        self.start_time = current_time
        self.reset(obs)
        
    def deactivate(self) -> None:
        """Deactivate this skill program."""
        self.active = False
        self.start_time = None
        
    def check_timeout(self, current_time: float) -> bool:
        """Check if timeout has been reached.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if timeout reached
        """
        if self.start_time is None:
            return False
        return (current_time - self.start_time) >= self.timeout
    
    def execute(self, obs: Dict[str, Any], current_time: float) -> SkillProgramResult:
        """Execute skill program with timeout checking.
        
        Args:
            obs: Current observation
            current_time: Current simulation time
            
        Returns:
            SkillProgramResult
        """
        # Check timeout
        timeout_reached = self.check_timeout(current_time)
        
        # Get policy output
        result = self.policy(obs)
        result.timeout_reached = timeout_reached
        
        # Check termination
        if self.should_terminate(obs) or timeout_reached:
            result.should_terminate = True
            if not result.success and timeout_reached:
                result.fallback_sp = self.get_fallback(obs)
        
        return result
