"""Self-play manager for curriculum learning.

This module manages opponent curriculum and self-play training.
"""
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    
    name: str
    stage_id: int
    min_timestep: int
    max_timestep: int
    game_mode: str  # "1v1", "2v2", "3v3"
    opponent_type: str  # "basic_script", "rule_policy", "selfplay", "checkpoint", "fast_opponent"
    difficulty: float = 0.5  # 0.0 to 1.0
    rotation_penalty_weight: float = 0.0  # Weight for rotation penalties
    speed_multiplier: float = 1.0  # Speed multiplier for opponents
    
    def is_active(self, timestep: int) -> bool:
        """Check if stage is active at given timestep."""
        return self.min_timestep <= timestep < self.max_timestep


class SelfPlayManager:
    """Manager for self-play curriculum and opponent pool."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize self-play manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.opponent_update_freq = self.config.get("opponent_update_freq", 100000)
        self.elo_threshold = self.config.get("elo_threshold", 100)
        
        # Create curriculum stages
        self.stages = self._create_curriculum_stages()
        self.current_stage_idx = 0
        
        # Opponent pool - stores previous checkpoints
        self.opponent_pool = []
        self.timesteps_since_update = 0
        
        # Statistics
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
        self.stage_matches = {i: {'wins': 0, 'losses': 0} for i in range(len(self.stages))}
        
        logger.info(f"SelfPlayManager initialized with {len(self.stages)} curriculum stages")
        for stage in self.stages:
            logger.info(f"  Stage {stage.stage_id}: {stage.name} ({stage.game_mode}, {stage.opponent_type})")
    
    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create comprehensive 9-stage curriculum with advanced mechanics.
        
        Returns:
            List of curriculum stages
        """
        stages = [
            # Stage 0: Basic 1v1 ground play
            CurriculumStage(
                name="Basic 1v1 Ground Play",
                stage_id=0,
                min_timestep=0,
                max_timestep=500_000,
                game_mode="1v1",
                opponent_type="basic_script",
                difficulty=0.15,
                rotation_penalty_weight=0.0,
                speed_multiplier=0.7
            ),
            # Stage 1: Boost management focus
            CurriculumStage(
                name="Boost Control & Management",
                stage_id=1,
                min_timestep=500_000,
                max_timestep=1_500_000,
                game_mode="1v1",
                opponent_type="rule_policy",
                difficulty=0.3,
                rotation_penalty_weight=0.0,
                speed_multiplier=0.9
            ),
            # Stage 2: Kickoff micro-strategy
            CurriculumStage(
                name="Kickoff Mastery",
                stage_id=2,
                min_timestep=1_500_000,
                max_timestep=2_500_000,
                game_mode="1v1",
                opponent_type="rule_policy",
                difficulty=0.4,
                rotation_penalty_weight=0.0,
                speed_multiplier=1.0
            ),
            # Stage 3: Aerial introduction
            CurriculumStage(
                name="Aerial Basics & Defense",
                stage_id=3,
                min_timestep=2_500_000,
                max_timestep=4_000_000,
                game_mode="1v1",
                opponent_type="selfplay",
                difficulty=0.5,
                rotation_penalty_weight=0.0,
                speed_multiplier=1.0
            ),
            # Stage 4: Advanced aerials
            CurriculumStage(
                name="Advanced Aerial Play",
                stage_id=4,
                min_timestep=4_000_000,
                max_timestep=5_500_000,
                game_mode="2v2",
                opponent_type="selfplay",
                difficulty=0.65,
                rotation_penalty_weight=0.2,
                speed_multiplier=1.0
            ),
            # Stage 5: 2v2 rotation and positioning
            CurriculumStage(
                name="2v2 Rotation Focus",
                stage_id=5,
                min_timestep=5_500_000,
                max_timestep=7_000_000,
                game_mode="2v2",
                opponent_type="selfplay",
                difficulty=0.7,
                rotation_penalty_weight=0.6,
                speed_multiplier=1.0
            ),
            # Stage 6: 1v2 defense scenarios
            CurriculumStage(
                name="1v2 Defensive Training",
                stage_id=6,
                min_timestep=7_000_000,
                max_timestep=8_500_000,
                game_mode="1v2",
                opponent_type="checkpoint",
                difficulty=0.8,
                rotation_penalty_weight=0.3,
                speed_multiplier=1.1
            ),
            # Stage 7: Fast-paced 3v3 
            CurriculumStage(
                name="3v3 Team Play",
                stage_id=7,
                min_timestep=8_500_000,
                max_timestep=10_000_000,
                game_mode="3v3",
                opponent_type="selfplay",
                difficulty=0.9,
                rotation_penalty_weight=0.7,
                speed_multiplier=1.0
            ),
            # Stage 8: Pro-level chaos
            CurriculumStage(
                name="Pro-Level 3v3 Chaos",
                stage_id=8,
                min_timestep=10_000_000,
                max_timestep=float('inf'),
                game_mode="3v3",
                opponent_type="checkpoint",
                difficulty=1.0,
                rotation_penalty_weight=0.8,
                speed_multiplier=1.15
            ),
        ]
        
        # Override with custom stages if provided
        custom_stages = self.config.get("custom_stages", [])
        if custom_stages:
            stages = self._parse_custom_stages(custom_stages)
        
        return stages
    
    def _parse_custom_stages(self, custom_config: List[Dict[str, Any]]) -> List[CurriculumStage]:
        """Parse custom curriculum stages from config.
        
        Args:
            custom_config: List of stage configurations
            
        Returns:
            List of CurriculumStage objects
        """
        stages = []
        for i, stage_cfg in enumerate(custom_config):
            stage = CurriculumStage(
                name=stage_cfg.get("name", f"Stage {i}"),
                stage_id=i,
                min_timestep=stage_cfg.get("min_timestep", 0),
                max_timestep=stage_cfg.get("max_timestep", float('inf')),
                game_mode=stage_cfg.get("game_mode", "1v1"),
                opponent_type=stage_cfg.get("opponent_type", "selfplay"),
                difficulty=stage_cfg.get("difficulty", 0.5),
                rotation_penalty_weight=stage_cfg.get("rotation_penalty_weight", 0.0),
                speed_multiplier=stage_cfg.get("speed_multiplier", 1.0)
            )
            stages.append(stage)
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
        
        # Return last stage if beyond all stages
        return self.stages[-1]
    
    def should_update_opponent(self, timesteps: int) -> bool:
        """Check if opponent should be updated.
        
        Args:
            timesteps: Current timestep count
            
        Returns:
            True if opponent should be updated
        """
        self.timesteps_since_update += 1
        
        if self.timesteps_since_update >= self.opponent_update_freq:
            self.timesteps_since_update = 0
            return True
        
        return False
    
    def should_transition_stage(self, timestep: int) -> Tuple[bool, Optional[CurriculumStage]]:
        """Check if we should transition to a new stage.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Tuple of (should_transition, new_stage)
        """
        current_stage = self.get_current_stage(timestep)
        
        # Check if we've moved to a new stage
        if current_stage.stage_id != self.current_stage_idx:
            self.current_stage_idx = current_stage.stage_id
            logger.info(
                f"Transitioning to stage {current_stage.stage_id}: {current_stage.name} "
                f"(timestep={timestep})"
            )
            return True, current_stage
        
        return False, None
    
    def add_opponent(self, model_path: Path, elo: float = 1500.0, timestep: int = 0):
        """Add opponent checkpoint to pool.
        
        Args:
            model_path: Path to opponent model checkpoint
            elo: Opponent Elo rating
            timestep: Timestep when checkpoint was saved
        """
        opponent_info = {
            "model_path": model_path,
            "elo": elo,
            "timestep": timestep,
            "wins": 0,
            "losses": 0,
            "last_used": 0
        }
        self.opponent_pool.append(opponent_info)
        logger.info(f"Added opponent to pool: {model_path} (Elo: {elo:.0f}, timestep: {timestep})")
    
    def select_opponent(
        self,
        current_stage: Optional[CurriculumStage] = None
    ) -> Optional[Dict[str, Any]]:
        """Select opponent for training based on current stage.
        
        Args:
            current_stage: Current curriculum stage
            
        Returns:
            Opponent info dict or None
        """
        if not current_stage:
            # Default to random selection from pool
            if not self.opponent_pool:
                return None
            return np.random.choice(self.opponent_pool)
        
        # Stage-specific opponent selection
        if current_stage.opponent_type == "basic_script":
            # Use basic scripted opponent (not from pool)
            return {"type": "basic_script", "model_path": None}
        
        elif current_stage.opponent_type == "rule_policy":
            # Use rule-based policy (not from pool)
            return {"type": "rule_policy", "model_path": None}
        
        elif current_stage.opponent_type in ["selfplay", "fast_opponent"]:
            # Select from opponent pool
            if not self.opponent_pool:
                logger.warning("No opponents in pool, using rule_policy as fallback")
                return {"type": "rule_policy", "model_path": None}
            
            # Select opponent with Elo closest to current agent
            # For simplicity, select randomly weighted by recency
            weights = np.array([1.0 / (1.0 + i) for i in range(len(self.opponent_pool))])
            weights /= weights.sum()
            
            opponent = np.random.choice(self.opponent_pool, p=weights)
            opponent["type"] = current_stage.opponent_type
            return opponent
        
        elif current_stage.opponent_type == "checkpoint":
            # Use specific checkpoint opponent
            if not self.opponent_pool:
                return None
            # Return most recent checkpoint
            return max(self.opponent_pool, key=lambda x: x["timestep"])
        
        else:
            logger.warning(f"Unknown opponent type: {current_stage.opponent_type}")
            return None
    
    def record_match_result(self, won: bool, stage_id: Optional[int] = None):
        """Record match result.
        
        Args:
            won: Whether agent won the match
            stage_id: ID of current stage (for stage-specific stats)
        """
        self.matches_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
        # Update stage-specific stats
        if stage_id is not None and stage_id in self.stage_matches:
            if won:
                self.stage_matches[stage_id]['wins'] += 1
            else:
                self.stage_matches[stage_id]['losses'] += 1
    
    def get_stage_config(self, timestep: int) -> Dict[str, Any]:
        """Get training configuration for current stage.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Configuration dictionary for current stage
        """
        stage = self.get_current_stage(timestep)
        
        return {
            'stage_id': stage.stage_id,
            'stage_name': stage.name,
            'game_mode': stage.game_mode,
            'opponent_type': stage.opponent_type,
            'difficulty': stage.difficulty,
            'rotation_penalty_weight': stage.rotation_penalty_weight,
            'speed_multiplier': stage.speed_multiplier
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get self-play statistics.
        
        Returns:
            Statistics dictionary
        """
        win_rate = self.wins / self.matches_played if self.matches_played > 0 else 0.0
        
        # Stage-specific win rates
        stage_stats = {}
        for stage_id, stats in self.stage_matches.items():
            total = stats['wins'] + stats['losses']
            stage_stats[f"stage_{stage_id}_winrate"] = (
                stats['wins'] / total if total > 0 else 0.0
            )
        
        return {
            "matches_played": self.matches_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "current_stage": self.current_stage_idx,
            "num_opponents_in_pool": len(self.opponent_pool),
            "stage_stats": stage_stats
        }
