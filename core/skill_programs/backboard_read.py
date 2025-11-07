"""Backboard Read Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_BackboardRead(SkillProgram):
    """Backboard read with double-tap viability check."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 2.0)
        
    def reset(self, obs: Dict[str, Any]) -> None:
        pass
    
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        # Implement backboard bounce prediction
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        return False
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_DoubleTap'
