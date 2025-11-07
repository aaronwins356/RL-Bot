"""Musty Flick Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_Musty(SkillProgram):
    """Musty flick mechanic.
    
    Execution: nose up 60-110°, late flip cancel backward to catapult ball.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 1.0)
        self.nose_up_angle_range = config.get('nose_angle', [60, 110])
        self.min_boost = config.get('min_boost', 30)
        
    def reset(self, obs: Dict[str, Any]) -> None:
        self.flip_executed = False
        
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        targets.trigger_musty = True
        
        # Implement musty flick execution
        # Nose up, then flip cancel backward
        
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        return self.flip_executed
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_AerialControl'
