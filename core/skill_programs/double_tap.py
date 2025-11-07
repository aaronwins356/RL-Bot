"""Double Tap Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_DoubleTap(SkillProgram):
    """Backboard double tap execution.
    
    Predict backboard impact P1 and post-bounce trajectory;
    pre-aim toward P2 with 150-250ms arrival buffer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 2.5)
        self.arrival_buffer_ms = config.get('arrival_buffer', [150, 250])
        self.aim_cone_deg = config.get('aim_cone_deg', 8)
        
    def reset(self, obs: Dict[str, Any]) -> None:
        self.first_touch = False
        self.second_touch = False
        
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        targets.trigger_double_tap = True
        
        # Implement double tap prediction and execution
        
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        return self.second_touch
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_BackboardRead'
