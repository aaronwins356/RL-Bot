"""Elo-based opponent sampling for self-play training.

This module implements intelligent opponent selection based on Elo ratings
to ensure optimal learning difficulty and progress.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpponentCheckpoint:
    """Opponent checkpoint with Elo rating."""
    
    checkpoint_path: str
    elo_rating: float
    timestep: int
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played


class EloBasedSampling:
    """Elo-based opponent sampling for optimal learning."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Elo-based sampling.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Sampling parameters
        self.temperature = self.config.get("sampling_temperature", 0.5)
        self.target_win_rate = self.config.get("target_win_rate", 0.5)
        self.elo_window = self.config.get("elo_window", 200)
        
        # Sampling strategies
        self.strategy = self.config.get("strategy", "elo_weighted")
        # Options: "uniform", "elo_weighted", "recent_weighted", "difficulty_adjusted"
        
        # Opponent pool
        self.opponent_pool: List[OpponentCheckpoint] = []
        self.current_elo = 1500.0
        
        logger.info(f"Elo-based sampling initialized (strategy={self.strategy})")
    
    def add_opponent(
        self,
        checkpoint_path: str,
        elo_rating: float,
        timestep: int
    ):
        """Add opponent checkpoint to pool.
        
        Args:
            checkpoint_path: Path to checkpoint
            elo_rating: Elo rating of checkpoint
            timestep: Training timestep when checkpoint was created
        """
        opponent = OpponentCheckpoint(
            checkpoint_path=checkpoint_path,
            elo_rating=elo_rating,
            timestep=timestep
        )
        self.opponent_pool.append(opponent)
        logger.info(f"Added opponent: Elo={elo_rating:.0f}, timestep={timestep}")
    
    def update_elo(self, new_elo: float):
        """Update current agent's Elo rating.
        
        Args:
            new_elo: New Elo rating
        """
        self.current_elo = new_elo
    
    def sample_opponent(self) -> Optional[OpponentCheckpoint]:
        """Sample an opponent from the pool.
        
        Returns:
            Selected opponent checkpoint or None if pool is empty
        """
        if len(self.opponent_pool) == 0:
            return None
        
        if self.strategy == "uniform":
            return self._sample_uniform()
        elif self.strategy == "elo_weighted":
            return self._sample_elo_weighted()
        elif self.strategy == "recent_weighted":
            return self._sample_recent_weighted()
        elif self.strategy == "difficulty_adjusted":
            return self._sample_difficulty_adjusted()
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using uniform")
            return self._sample_uniform()
    
    def _sample_uniform(self) -> OpponentCheckpoint:
        """Sample opponent uniformly."""
        return np.random.choice(self.opponent_pool)
    
    def _sample_elo_weighted(self) -> OpponentCheckpoint:
        """Sample opponent weighted by Elo proximity.
        
        Prefers opponents with similar Elo ratings (within window).
        """
        # Compute weights based on Elo difference
        weights = []
        for opp in self.opponent_pool:
            elo_diff = abs(opp.elo_rating - self.current_elo)
            # Higher weight for closer Elo ratings
            weight = np.exp(-elo_diff / self.elo_window)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample
        idx = np.random.choice(len(self.opponent_pool), p=weights)
        return self.opponent_pool[idx]
    
    def _sample_recent_weighted(self) -> OpponentCheckpoint:
        """Sample opponent weighted by recency.
        
        Prefers more recent checkpoints.
        """
        # Get timesteps
        timesteps = np.array([opp.timestep for opp in self.opponent_pool])
        max_timestep = timesteps.max()
        
        # Compute weights (exponential decay from most recent)
        weights = np.exp((timesteps - max_timestep) / 1_000_000 * self.temperature)
        weights = weights / weights.sum()
        
        # Sample
        idx = np.random.choice(len(self.opponent_pool), p=weights)
        return self.opponent_pool[idx]
    
    def _sample_difficulty_adjusted(self) -> OpponentCheckpoint:
        """Sample opponent based on target win rate.
        
        Adjusts selection to maintain target win rate (~50%).
        """
        # Estimate expected score against each opponent
        expected_scores = []
        for opp in self.opponent_pool:
            elo_diff = self.current_elo - opp.elo_rating
            expected = 1 / (1 + 10 ** (-elo_diff / 400))
            expected_scores.append(expected)
        
        expected_scores = np.array(expected_scores)
        
        # Weight by distance from target win rate
        target_distance = np.abs(expected_scores - self.target_win_rate)
        weights = np.exp(-target_distance / self.temperature)
        weights = weights / weights.sum()
        
        # Sample
        idx = np.random.choice(len(self.opponent_pool), p=weights)
        return self.opponent_pool[idx]
    
    def update_opponent_stats(
        self,
        opponent: OpponentCheckpoint,
        won: bool
    ):
        """Update opponent statistics after a game.
        
        Args:
            opponent: Opponent that was played
            won: Whether agent won the game
        """
        opponent.games_played += 1
        if won:
            opponent.wins += 1
        else:
            opponent.losses += 1
    
    def get_opponent_distribution(self) -> Dict[str, Any]:
        """Get statistics about opponent pool.
        
        Returns:
            Dictionary with pool statistics
        """
        if len(self.opponent_pool) == 0:
            return {"count": 0}
        
        elo_ratings = [opp.elo_rating for opp in self.opponent_pool]
        timesteps = [opp.timestep for opp in self.opponent_pool]
        
        return {
            "count": len(self.opponent_pool),
            "elo_mean": np.mean(elo_ratings),
            "elo_std": np.std(elo_ratings),
            "elo_min": np.min(elo_ratings),
            "elo_max": np.max(elo_ratings),
            "timestep_range": (np.min(timesteps), np.max(timesteps)),
            "current_elo": self.current_elo
        }
    
    def prune_weak_opponents(self, min_elo_diff: float = 300):
        """Remove opponents that are too weak.
        
        Args:
            min_elo_diff: Minimum Elo difference to keep opponent
        """
        original_count = len(self.opponent_pool)
        
        self.opponent_pool = [
            opp for opp in self.opponent_pool
            if self.current_elo - opp.elo_rating < min_elo_diff
        ]
        
        pruned = original_count - len(self.opponent_pool)
        if pruned > 0:
            logger.info(f"Pruned {pruned} weak opponents from pool")
    
    def get_top_opponents(self, n: int = 5) -> List[OpponentCheckpoint]:
        """Get top N opponents by Elo.
        
        Args:
            n: Number of top opponents to return
            
        Returns:
            List of top opponents
        """
        sorted_opponents = sorted(
            self.opponent_pool,
            key=lambda x: x.elo_rating,
            reverse=True
        )
        return sorted_opponents[:n]
