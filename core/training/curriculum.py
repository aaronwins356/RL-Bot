"""Curriculum learning for RL-Bot training.

This module implements curriculum training strategies, including aerial-focused training.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    
    name: str
    min_timesteps: int
    max_timesteps: int
    difficulty: float  # 0.0 to 1.0
    aerial_focus: bool = False
    game_mode: str = "1v1"  # "1v1", "2v2", "3v3"
    
    def is_active(self, current_timestep: int) -> bool:
        """Check if this stage is active at current timestep.
        
        Args:
            current_timestep: Current training timestep
            
        Returns:
            True if stage is active
        """
        return self.min_timesteps <= current_timestep < self.max_timesteps


class CurriculumManager:
    """Manager for curriculum learning strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize curriculum manager.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config
        self.aerial_focus = config.get('aerial_focus', False)
        self.stages = self._create_stages()
        self.current_stage_idx = 0
        
        logger.info(f"Curriculum manager initialized with {len(self.stages)} stages")
        if self.aerial_focus:
            logger.info("Aerial-focused curriculum enabled")
    
    def _create_stages(self) -> List[CurriculumStage]:
        """Create curriculum stages based on config.
        
        Returns:
            List of curriculum stages
        """
        stages = []
        
        if self.aerial_focus:
            # Aerial-focused curriculum
            stages = [
                CurriculumStage(
                    name="Basic Ground Skills",
                    min_timesteps=0,
                    max_timesteps=1_000_000,
                    difficulty=0.2,
                    aerial_focus=False,
                    game_mode="1v1"
                ),
                CurriculumStage(
                    name="Aerial Introduction",
                    min_timesteps=1_000_000,
                    max_timesteps=3_000_000,
                    difficulty=0.4,
                    aerial_focus=True,
                    game_mode="1v1"
                ),
                CurriculumStage(
                    name="Advanced Aerials",
                    min_timesteps=3_000_000,
                    max_timesteps=7_000_000,
                    difficulty=0.7,
                    aerial_focus=True,
                    game_mode="2v2"
                ),
                CurriculumStage(
                    name="Master Play",
                    min_timesteps=7_000_000,
                    max_timesteps=float('inf'),
                    difficulty=1.0,
                    aerial_focus=False,
                    game_mode="3v3"
                ),
            ]
        else:
            # Standard progressive curriculum
            stages = [
                CurriculumStage(
                    name="1v1 Basics",
                    min_timesteps=0,
                    max_timesteps=2_000_000,
                    difficulty=0.3,
                    game_mode="1v1"
                ),
                CurriculumStage(
                    name="2v2 Intermediate",
                    min_timesteps=2_000_000,
                    max_timesteps=5_000_000,
                    difficulty=0.6,
                    game_mode="2v2"
                ),
                CurriculumStage(
                    name="3v3 Advanced",
                    min_timesteps=5_000_000,
                    max_timesteps=float('inf'),
                    difficulty=0.9,
                    game_mode="3v3"
                ),
            ]
        
        return stages
    
    def get_current_stage(self, timestep: int) -> CurriculumStage:
        """Get current curriculum stage for given timestep.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Current curriculum stage
        """
        for stage in self.stages:
            if stage.is_active(timestep):
                return stage
        
        # Default to last stage
        return self.stages[-1]
    
    def should_transition(self, timestep: int) -> bool:
        """Check if we should transition to next stage.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            True if should transition to next stage
        """
        current_stage = self.get_current_stage(timestep)
        
        # Check if we've moved past current stage
        if timestep >= current_stage.max_timesteps:
            next_stage_idx = self.current_stage_idx + 1
            if next_stage_idx < len(self.stages):
                self.current_stage_idx = next_stage_idx
                logger.info(
                    f"Transitioning to stage {self.current_stage_idx}: "
                    f"{self.stages[self.current_stage_idx].name}"
                )
                return True
        
        return False
    
    def get_training_config(self, timestep: int) -> Dict[str, Any]:
        """Get training configuration for current stage.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Training configuration dict
        """
        stage = self.get_current_stage(timestep)
        
        config = {
            'stage_name': stage.name,
            'difficulty': stage.difficulty,
            'aerial_focus': stage.aerial_focus,
            'game_mode': stage.game_mode,
        }
        
        # Aerial-specific modifications
        if stage.aerial_focus:
            config['spawn_height_range'] = (200, 800)  # Higher spawns
            config['boost_amount_multiplier'] = 1.5  # More boost available
            config['aerial_reward_weight'] = 2.0  # Emphasize aerial rewards
        
        return config
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'current_stage': self.current_stage_idx,
            'total_stages': len(self.stages),
            'aerial_focus': self.aerial_focus
        }
