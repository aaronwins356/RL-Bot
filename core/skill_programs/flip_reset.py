"""Flip Reset Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_FlipReset(SkillProgram):
    """Flip reset detection and execution."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 1.5)
        self.contact_window_ms = config.get('contact_window_ms', 20)
        self.flip_delay_max_ms = config.get('flip_delay_max_ms', 400)
        
    def reset(self, obs: Dict[str, Any]) -> None:
        self.flip_stored = False
        self.contact_time = None
        
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        targets.trigger_flip_reset = True
        
        # Implement flip reset contact planning
        # Throttle feather + minimal boost for low relative velocity
        
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        return self.flip_stored
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_AerialControl'
