"""Ceiling Shot Skill Programs."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_CeilingSetup(SkillProgram):
    """Setup for ceiling shot: carry ball to ceiling."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 3.0)
        self.target_speed_range = config.get('target_speed', [1200, 1500])
        self.takeoff_angle_range = config.get('takeoff_angle', [30, 45])
        
    def reset(self, obs: Dict[str, Any]) -> None:
        pass
    
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        # Implement ceiling setup logic
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        # Check if at ceiling with jump reset available
        car_pos = obs.get('car_position', np.zeros(3))
        return car_pos[2] > 1900  # Near ceiling
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_AerialControl'


class SP_CeilingShot(SkillProgram):
    """Execute ceiling shot after setup."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 2.0)
        
    def reset(self, obs: Dict[str, Any]) -> None:
        pass
    
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        targets.trigger_ceiling_setup = True
        # Implement ceiling shot execution
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        return False
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_BackboardRead'
