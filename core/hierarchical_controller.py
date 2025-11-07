"""Hierarchical Controller that coordinates OD, SPs, and LLC.

This is the main entry point for the hierarchical RL/IL system.
"""

from typing import Dict, Any, Optional
import numpy as np
import time

from core.opportunity_detector import OpportunityDetector, GameStateCategory, RiskScorer
from core.skill_programs import (
    SkillProgram, SP_FastAerial, SP_AerialControl, SP_WallRead,
    SP_BackboardRead, SP_CeilingSetup, SP_CeilingShot, SP_FlipReset,
    SP_Musty, SP_Breezi, SP_DoubleTap, SP_GroundToAirDribble
)
from core.llc import LowLevelController


class HierarchicalController:
    """Three-layer hierarchical controller.
    
    Layer A: Opportunity Detector (OD) - Classifies game state and selects SP
    Layer B: Skill Programs (SP) - Modular micro-policies for each mechanic
    Layer C: Low-Level Controller (LLC) - Converts SP targets to control surfaces
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """Initialize hierarchical controller.
        
        Args:
            config: Configuration dict
            device: Device for neural networks
        """
        self.config = config
        self.device = device
        
        # Layer A: Opportunity Detector
        od_config = config.get('opportunity_detector', {})
        self.opportunity_detector = OpportunityDetector(od_config, device)
        self.risk_scorer = RiskScorer(od_config.get('risk_scorer', {}))
        
        # Layer B: Skill Programs
        controller_config = config.get('controllers', {})
        self.skill_programs: Dict[str, SkillProgram] = {
            'SP_FastAerial': SP_FastAerial(controller_config.get('fast_aerial', {})),
            'SP_AerialControl': SP_AerialControl(controller_config.get('aerial_control', {})),
            'SP_WallRead': SP_WallRead(controller_config.get('wall_read', {})),
            'SP_BackboardRead': SP_BackboardRead(controller_config.get('backboard_read', {})),
            'SP_CeilingSetup': SP_CeilingSetup(controller_config.get('ceiling', {})),
            'SP_CeilingShot': SP_CeilingShot(controller_config.get('ceiling', {})),
            'SP_FlipReset': SP_FlipReset(controller_config.get('flip_reset', {})),
            'SP_Musty': SP_Musty(controller_config.get('musty', {})),
            'SP_Breezi': SP_Breezi(controller_config.get('breezi', {})),
            'SP_DoubleTap': SP_DoubleTap(controller_config.get('double_tap', {})),
            'SP_GroundToAirDribble': SP_GroundToAirDribble(controller_config.get('ground_to_air_dribble', {})),
        }
        
        # Layer C: Low-Level Controller
        llc_config = controller_config.get('llc', {})
        self.llc = LowLevelController(llc_config)
        
        # Current state
        self.current_sp: Optional[SkillProgram] = None
        self.current_sp_name: Optional[str] = None
        self.current_category: Optional[GameStateCategory] = None
        self.frame_count = 0
        
        # Feature flags
        self.flashy_enabled = config.get('features', {}).get('flashy_enabled', False)
        
        # Telemetry
        self.telemetry = {
            'sp_choices': [],
            'sp_successes': [],
            'sp_timeouts': [],
            'confidence_history': [],
            'risk_scores': [],
        }
        
    def reset(self):
        """Reset controller state."""
        self.opportunity_detector.reset()
        self.llc.reset()
        self.current_sp = None
        self.current_sp_name = None
        self.current_category = None
        self.frame_count = 0
        
        # Reset all SPs
        for sp in self.skill_programs.values():
            sp.deactivate()
    
    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from hierarchical controller.
        
        Args:
            obs: Current observation
            
        Returns:
            Action dict with control values
        """
        current_time = obs.get('time', 0.0)
        self.frame_count += 1
        
        # Compute risk score
        risk_score = self.risk_scorer.compute_risk_score(obs)
        self.telemetry['risk_scores'].append(risk_score)
        
        # Layer A: Opportunity Detection
        # Only re-detect if no active SP or current SP should terminate
        if self.current_sp is None or self.current_sp.should_terminate(obs):
            # Deactivate current SP if it exists
            if self.current_sp is not None:
                self.current_sp.deactivate()
                
                # Update bandit statistics
                sp_result = self.current_sp.execute(obs, current_time)
                self.opportunity_detector.update_bandit(
                    self.current_sp_name, 
                    sp_result.success
                )
                
                # Log telemetry
                self.telemetry['sp_successes'].append({
                    'sp': self.current_sp_name,
                    'success': sp_result.success,
                    'frame': self.frame_count,
                })
                if sp_result.timeout_reached:
                    self.telemetry['sp_timeouts'].append({
                        'sp': self.current_sp_name,
                        'frame': self.frame_count,
                    })
            
            # Detect new opportunity and select SP
            category, sp_name, confidence = self.opportunity_detector.detect(obs, risk_score)
            
            # Filter flashy if not enabled or risk too high
            if not self.flashy_enabled and category == GameStateCategory.FLASHY_OPPORTUNITY:
                category = GameStateCategory.AERIAL_OPPORTUNITY
                sp_name = 'SP_AerialControl'
            
            # Check if should attempt flashy based on heuristics
            if category == GameStateCategory.FLASHY_OPPORTUNITY:
                if not self.risk_scorer.should_attempt_flashy(obs):
                    category = GameStateCategory.AERIAL_OPPORTUNITY
                    sp_name = 'SP_AerialControl'
            
            # Activate new SP
            self.current_category = category
            self.current_sp_name = sp_name
            self.current_sp = self.skill_programs[sp_name]
            self.current_sp.activate(obs, current_time)
            
            # Log telemetry
            self.telemetry['sp_choices'].append({
                'sp': sp_name,
                'category': category.name,
                'confidence': confidence,
                'risk_score': risk_score,
                'frame': self.frame_count,
            })
            self.telemetry['confidence_history'].append(confidence)
        
        # Layer B: Execute Skill Program
        sp_result = self.current_sp.execute(obs, current_time)
        
        # Check for fallback
        if sp_result.should_terminate and sp_result.fallback_sp:
            fallback_sp_name = sp_result.fallback_sp
            if fallback_sp_name in self.skill_programs:
                self.current_sp.deactivate()
                self.current_sp_name = fallback_sp_name
                self.current_sp = self.skill_programs[fallback_sp_name]
                self.current_sp.activate(obs, current_time)
                
                # Re-execute with fallback SP
                sp_result = self.current_sp.execute(obs, current_time)
        
        # Layer C: Low-Level Controller
        controls = self.llc.compute_controls(sp_result.targets, obs, self.frame_count)
        
        # Add metadata for reward computation
        controls['_metadata'] = {
            'sp_name': self.current_sp_name,
            'category': self.current_category.name if self.current_category else 'unknown',
            'confidence': self.telemetry['confidence_history'][-1] if self.telemetry['confidence_history'] else 0.0,
            'risk_score': risk_score,
            'flashy_ok': self.current_category == GameStateCategory.FLASHY_OPPORTUNITY,
            'sp_success': sp_result.success,
            'sp_metrics': sp_result.metrics,
        }
        
        return controls
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data.
        
        Returns:
            Telemetry dict
        """
        return self.telemetry.copy()
    
    def load_od_checkpoint(self, path: str):
        """Load opportunity detector checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        self.opportunity_detector.load_checkpoint(path)
    
    def save_od_checkpoint(self, path: str):
        """Save opportunity detector checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        self.opportunity_detector.save_checkpoint(path)
    
    def enable_flashy_mechanics(self, enabled: bool = True):
        """Enable or disable flashy mechanics.
        
        Args:
            enabled: Whether to enable flashy mechanics
        """
        self.flashy_enabled = enabled
