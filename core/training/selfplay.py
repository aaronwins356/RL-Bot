"""Self-play manager for curriculum learning.

This module manages opponent curriculum and self-play training.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np


class SelfPlayManager:
    """Manager for self-play curriculum and opponent pool."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize self-play manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.curriculum_stages = self.config.get("curriculum_stages", ["1v1"])
        self.opponent_update_freq = self.config.get("opponent_update_freq", 100000)
        self.elo_threshold = self.config.get("elo_threshold", 100)
        
        # Opponent pool
        self.opponents = []
        self.current_stage = 0
        self.timesteps_since_update = 0
        
        # Statistics
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
    
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
    
    def add_opponent(self, model_path: Path, elo: float = 1500.0):
        """Add opponent to pool.
        
        Args:
            model_path: Path to opponent model
            elo: Opponent Elo rating
        """
        self.opponents.append({
            "model_path": model_path,
            "elo": elo,
            "wins": 0,
            "losses": 0
        })
    
    def select_opponent(self) -> Optional[Dict[str, Any]]:
        """Select opponent for training.
        
        Returns:
            Opponent info or None
        """
        if not self.opponents:
            return None
        
        # Select opponent based on Elo (closer to current agent's Elo)
        # For simplicity, select randomly for now
        return np.random.choice(self.opponents)
    
    def record_match_result(self, won: bool):
        """Record match result.
        
        Args:
            won: Whether agent won the match
        """
        self.matches_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get self-play statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "matches_played": self.matches_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / self.matches_played if self.matches_played > 0 else 0.0,
            "current_stage": self.current_stage,
            "num_opponents": len(self.opponents)
        }
