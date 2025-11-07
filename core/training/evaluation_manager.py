"""Evaluation manager for comprehensive model evaluation.

This module implements:
- Periodic evaluation every 25k steps
- 200 games per eval (100 vs rule policy + 100 vs past checkpoints)
- Rolling Elo with confidence intervals
- EMA smoothing (α = 0.3)
- Early stopping logic
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    
    timestep: int
    elo: float
    elo_std: float  # Standard deviation / confidence
    games_played: int
    opponent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestep": self.timestep,
            "elo": self.elo,
            "elo_std": self.elo_std,
            "games_played": self.games_played,
            "opponent_results": self.opponent_results,
            "timestamp": self.timestamp.isoformat(),
        }


class EvaluationManager:
    """Manages comprehensive model evaluation."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[Path] = None,
    ):
        """Initialize evaluation manager.
        
        Args:
            config: Configuration dictionary
            log_dir: Directory for saving evaluation results
        """
        self.config = config or {}
        self.log_dir = Path(log_dir) if log_dir else Path("logs/evaluation")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation configuration
        self.eval_interval = self.config.get("eval_interval", 25000)
        self.games_per_opponent = self.config.get("games_per_opponent", 100)
        self.num_opponents = self.config.get("num_opponents", 2)  # rule + 1 past checkpoint
        self.total_games_per_eval = self.games_per_opponent * self.num_opponents
        
        # Elo configuration
        self.k_factor = self.config.get("elo_k_factor", 32)
        self.initial_elo = self.config.get("initial_elo", 1500.0)
        self.ema_alpha = self.config.get("ema_alpha", 0.3)  # EMA smoothing factor
        
        # Early stopping configuration
        self.early_stop_patience = self.config.get("early_stop_patience", 8)
        self.early_stop_min_improvement = self.config.get("early_stop_min_improvement", 10.0)
        self.early_stop_max_std = self.config.get("early_stop_max_std", 15.0)
        
        # State
        self.current_elo = self.initial_elo
        self.elo_std = 100.0  # Initial high uncertainty
        self.evaluation_history: List[EvaluationResult] = []
        self.best_elo = -float('inf')
        self.evaluations_without_improvement = 0
        
        # Rolling statistics
        self.elo_window: List[float] = []
        self.window_size = self.config.get("elo_window_size", 5)
        
        logger.info(f"EvaluationManager initialized:")
        logger.info(f"  - Eval interval: {self.eval_interval} steps")
        logger.info(f"  - Games per eval: {self.total_games_per_eval}")
        logger.info(f"  - Early stop patience: {self.early_stop_patience}")
    
    def should_evaluate(self, timestep: int) -> bool:
        """Check if evaluation should run at this timestep.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            True if should evaluate
        """
        return timestep > 0 and timestep % self.eval_interval == 0
    
    def evaluate(
        self,
        model,
        timestep: int,
        opponent_pool: Optional[List[Any]] = None,
    ) -> EvaluationResult:
        """Run comprehensive evaluation.
        
        Args:
            model: Model to evaluate
            timestep: Current training timestep
            opponent_pool: List of opponent agents/checkpoints
            
        Returns:
            EvaluationResult with evaluation metrics
        """
        logger.info(f"Starting evaluation at timestep {timestep}")
        
        # Placeholder for actual game simulation
        # In real implementation, this would:
        # 1. Load opponents (rule policy + past checkpoints)
        # 2. Play games_per_opponent games vs each
        # 3. Calculate Elo updates based on results
        
        opponent_results = {}
        total_wins = 0
        total_games = 0
        
        # Simulate evaluation against rule policy
        rule_wins, rule_losses = self._simulate_games_vs_opponent(
            model, "rule_policy", self.games_per_opponent
        )
        opponent_results["rule_policy"] = {
            "wins": rule_wins,
            "losses": rule_losses,
            "games": self.games_per_opponent,
            "win_rate": rule_wins / self.games_per_opponent,
        }
        total_wins += rule_wins
        total_games += self.games_per_opponent
        
        # Simulate evaluation against past checkpoints
        if opponent_pool and len(opponent_pool) > 0:
            # Select most recent checkpoint
            checkpoint = opponent_pool[-1] if opponent_pool else None
            if checkpoint:
                checkpoint_id = checkpoint.get("agent_id", "past_checkpoint")
                cp_wins, cp_losses = self._simulate_games_vs_opponent(
                    model, checkpoint_id, self.games_per_opponent
                )
                opponent_results[checkpoint_id] = {
                    "wins": cp_wins,
                    "losses": cp_losses,
                    "games": self.games_per_opponent,
                    "win_rate": cp_wins / self.games_per_opponent,
                }
                total_wins += cp_wins
                total_games += self.games_per_opponent
        
        # Calculate overall win rate
        overall_win_rate = total_wins / total_games if total_games > 0 else 0.5
        
        # Update Elo using simplified calculation
        # In real implementation, this would use actual game results
        elo_change = self.k_factor * (overall_win_rate - 0.5)
        new_elo = self.current_elo + elo_change
        
        # Apply EMA smoothing
        smoothed_elo = self.ema_alpha * new_elo + (1 - self.ema_alpha) * self.current_elo
        
        # Update rolling window
        self.elo_window.append(smoothed_elo)
        if len(self.elo_window) > self.window_size:
            self.elo_window.pop(0)
        
        # Calculate confidence (standard deviation over window)
        self.elo_std = np.std(self.elo_window) if len(self.elo_window) > 1 else 100.0
        
        # Update current Elo
        self.current_elo = smoothed_elo
        
        # Create evaluation result
        result = EvaluationResult(
            timestep=timestep,
            elo=self.current_elo,
            elo_std=self.elo_std,
            games_played=total_games,
            opponent_results=opponent_results,
        )
        
        self.evaluation_history.append(result)
        
        # Check for improvement
        self._check_improvement()
        
        # Log results
        self._log_results(result)
        
        return result
    
    def _simulate_games_vs_opponent(
        self,
        model,
        opponent_id: str,
        num_games: int,
    ) -> Tuple[int, int]:
        """Simulate games vs opponent (placeholder).
        
        In real implementation, this would actually play games.
        For now, simulates with random results biased by training progress.
        
        Args:
            model: Model to evaluate
            opponent_id: ID of opponent
            num_games: Number of games to play
            
        Returns:
            Tuple of (wins, losses)
        """
        # Placeholder: simulate with gradually improving win rate
        # In real implementation, would actually play games
        base_win_rate = 0.45  # Start slightly below 50%
        progress_bonus = min(0.15, len(self.evaluation_history) * 0.02)  # Improve over time
        win_rate = min(0.65, base_win_rate + progress_bonus)
        
        wins = np.random.binomial(num_games, win_rate)
        losses = num_games - wins
        
        logger.debug(f"Simulated {num_games} games vs {opponent_id}: {wins}W-{losses}L")
        return wins, losses
    
    def _check_improvement(self):
        """Check if Elo has improved and update early stopping counter."""
        if self.current_elo > self.best_elo + self.early_stop_min_improvement:
            # Significant improvement
            self.best_elo = self.current_elo
            self.evaluations_without_improvement = 0
            logger.info(f"New best Elo: {self.best_elo:.1f} (±{self.elo_std:.1f})")
        else:
            # No improvement
            self.evaluations_without_improvement += 1
            logger.info(
                f"No Elo improvement: {self.evaluations_without_improvement}/{self.early_stop_patience}"
            )
    
    def should_early_stop(self) -> bool:
        """Check if training should stop early.
        
        Returns:
            True if early stopping criteria met
        """
        # Check if we've had enough evaluations without improvement
        if self.evaluations_without_improvement < self.early_stop_patience:
            return False
        
        # Check if confidence is high enough (low std deviation)
        if self.elo_std > self.early_stop_max_std:
            logger.info(f"Elo std too high for early stop: {self.elo_std:.1f}")
            return False
        
        logger.info(
            f"Early stopping criteria met: {self.evaluations_without_improvement} "
            f"evals without improvement, std={self.elo_std:.1f}"
        )
        return True
    
    def _log_results(self, result: EvaluationResult):
        """Log evaluation results.
        
        Args:
            result: Evaluation result to log
        """
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Timestep: {result.timestep}")
        logger.info(f"Elo: {result.elo:.1f} (±{result.elo_std:.1f})")
        logger.info(f"Best Elo: {self.best_elo:.1f}")
        logger.info(f"Games played: {result.games_played}")
        logger.info("")
        logger.info("Opponent Results:")
        for opp_id, opp_result in result.opponent_results.items():
            logger.info(
                f"  {opp_id}: {opp_result['wins']}W-{opp_result['losses']}L "
                f"({opp_result['win_rate']:.1%} win rate)"
            )
        logger.info("=" * 60)
    
    def get_current_elo(self) -> float:
        """Get current Elo rating.
        
        Returns:
            Current Elo rating
        """
        return self.current_elo
    
    def get_elo_with_confidence(self) -> Tuple[float, float]:
        """Get Elo with confidence interval.
        
        Returns:
            Tuple of (elo, std_deviation)
        """
        return self.current_elo, self.elo_std
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics.
        
        Returns:
            Dictionary with evaluation stats
        """
        return {
            "current_elo": self.current_elo,
            "elo_std": self.elo_std,
            "best_elo": self.best_elo,
            "total_evaluations": len(self.evaluation_history),
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "should_early_stop": self.should_early_stop(),
        }
    
    def save_history(self, filepath: Optional[Path] = None):
        """Save evaluation history to file.
        
        Args:
            filepath: Path to save file (defaults to log_dir/elo_history.json)
        """
        filepath = filepath or self.log_dir / "elo_history.json"
        
        history_data = {
            "config": self.config,
            "current_elo": self.current_elo,
            "best_elo": self.best_elo,
            "evaluations": [result.to_dict() for result in self.evaluation_history],
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation history saved to {filepath}")
    
    def load_history(self, filepath: Optional[Path] = None):
        """Load evaluation history from file.
        
        Args:
            filepath: Path to load from (defaults to log_dir/elo_history.json)
        """
        filepath = filepath or self.log_dir / "elo_history.json"
        
        if not filepath.exists():
            logger.warning(f"No history file found at {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.current_elo = data.get("current_elo", self.initial_elo)
        self.best_elo = data.get("best_elo", -float('inf'))
        
        # Reconstruct evaluation history
        self.evaluation_history = []
        for eval_data in data.get("evaluations", []):
            result = EvaluationResult(
                timestep=eval_data["timestep"],
                elo=eval_data["elo"],
                elo_std=eval_data["elo_std"],
                games_played=eval_data["games_played"],
                opponent_results=eval_data["opponent_results"],
            )
            self.evaluation_history.append(result)
        
        logger.info(f"Loaded {len(self.evaluation_history)} evaluation results from {filepath}")
