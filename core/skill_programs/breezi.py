"""Breezi Skill Program."""

from typing import Dict, Any, Optional
import numpy as np
from .base import SkillProgram, LowLevelTargets, SkillProgramResult


class SP_Breezi(SkillProgram):
    """Breezi mechanic with oscillatory air-roll.
    
    Execution: oscillatory air-roll + yaw micro-adjusts to induce lateral spin.
    Requires boost ≥ 40, approach angle 20-35°.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('timeout', 1.5)
        self.roll_freq_hz = config.get('roll_freq_hz', [5, 9])
        self.roll_amp = config.get('roll_amp', [0.12, 0.25])
        self.min_boost = config.get('min_boost', 40)
        self.approach_angle_range = config.get('approach_angle', [20, 35])
        
    def reset(self, obs: Dict[str, Any]) -> None:
        self.start_time = obs.get('time', 0.0)
        
    def policy(self, obs: Dict[str, Any]) -> SkillProgramResult:
        targets = LowLevelTargets()
        targets.trigger_breezi = True
        
        current_time = obs.get('time', 0.0)
        elapsed = current_time - self.start_time
        
        # Oscillatory air roll
        freq = (self.roll_freq_hz[0] + self.roll_freq_hz[1]) / 2
        amp = (self.roll_amp[0] + self.roll_amp[1]) / 2
        targets.roll = amp * np.sin(2 * np.pi * freq * elapsed)
        
        return SkillProgramResult(targets=targets)
    
    def should_terminate(self, obs: Dict[str, Any]) -> bool:
        return False
    
    def get_fallback(self, obs: Dict[str, Any]) -> Optional[str]:
        return 'SP_AerialControl'
