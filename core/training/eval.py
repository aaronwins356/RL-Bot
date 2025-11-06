"""Evaluation module for computing Elo ratings and generating plots.

This module provides evaluation against scripted baselines.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import csv
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class EloEvaluator:
    """Evaluator for computing Elo ratings vs baselines."""
    
    # Elo calculation constants
    K_FACTOR = 32
    INITIAL_ELO = 1500
    
    def __init__(
        self,
        baseline_bots: Optional[List[str]] = None,
        log_dir: Optional[Path] = None
    ):
        """Initialize evaluator.
        
        Args:
            baseline_bots: List of baseline bot names
            log_dir: Directory for saving eval logs
        """
        self.baseline_bots = baseline_bots or ["rookie", "allstar", "nexto"]
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Elo ratings
        self.agent_elo = self.INITIAL_ELO
        self.baseline_elos = {
            "rookie": 800,
            "allstar": 1500,
            "nexto": 2000,
            "rule_policy": 1200,
            "previous_checkpoint": self.INITIAL_ELO
        }
        
        # Match history
        self.match_history = []
        
        # Summary CSV
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.summary_csv = self.log_dir / "eval_summary.csv"
            self._init_summary_csv()
    
    def _init_summary_csv(self):
        """Initialize eval summary CSV file."""
        if not self.summary_csv.exists():
            with open(self.summary_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'timestep',
                    'opponent',
                    'wins',
                    'losses',
                    'win_rate',
                    'elo_before',
                    'elo_after',
                    'elo_change'
                ])
    
    def evaluate_full(
        self,
        model,
        timestep: int,
        num_games: int = 5
    ) -> Dict[str, Any]:
        """Run full evaluation suite.
        
        Args:
            model: Model to evaluate
            timestep: Current training timestep
            num_games: Number of games per opponent
            
        Returns:
            Evaluation results dictionary
        """
        results = {}
        
        # Evaluate vs RulePolicy
        logger.info(f"Evaluating vs RulePolicy ({num_games} games)...")
        rule_wins, rule_losses = self._play_matches(
            model, "rule_policy", num_games
        )
        rule_elo = self.evaluate_against_baseline(
            rule_wins, rule_losses, "rule_policy"
        )
        results['rule_policy'] = {
            'wins': rule_wins,
            'losses': rule_losses,
            'win_rate': rule_wins / (rule_wins + rule_losses) if (rule_wins + rule_losses) > 0 else 0.0,
            'elo': rule_elo
        }
        
        # Evaluate vs previous checkpoint
        logger.info(f"Evaluating vs previous checkpoint ({num_games} games)...")
        prev_wins, prev_losses = self._play_matches(
            model, "previous_checkpoint", num_games
        )
        prev_elo = self.evaluate_against_baseline(
            prev_wins, prev_losses, "previous_checkpoint"
        )
        results['previous_checkpoint'] = {
            'wins': prev_wins,
            'losses': prev_losses,
            'win_rate': prev_wins / (prev_wins + prev_losses) if (prev_wins + prev_losses) > 0 else 0.0,
            'elo': prev_elo
        }
        
        # Log results
        logger.info(f"Evaluation complete:")
        logger.info(f"  vs RulePolicy: {rule_wins}W-{rule_losses}L "
                   f"({results['rule_policy']['win_rate']:.1%})")
        logger.info(f"  vs Previous: {prev_wins}W-{prev_losses}L "
                   f"({results['previous_checkpoint']['win_rate']:.1%})")
        logger.info(f"  Current Elo: {self.agent_elo:.0f}")
        
        # Save to CSV
        self._save_eval_results(timestep, results)
        
        return results
    
    def _play_matches(
        self,
        model,
        opponent: str,
        num_games: int
    ) -> Tuple[int, int]:
        """Play matches against opponent.
        
        Args:
            model: Model to evaluate
            opponent: Opponent name
            num_games: Number of games to play
            
        Returns:
            Tuple of (wins, losses)
        """
        # Placeholder for actual match playing
        # In real implementation, this would run games in RLGym/RLBot
        
        # Simulate random results for now
        wins = np.random.randint(0, num_games + 1)
        losses = num_games - wins
        
        return wins, losses
    
    def _save_eval_results(self, timestep: int, results: Dict[str, Any]):
        """Save evaluation results to CSV.
        
        Args:
            timestep: Current timestep
            results: Evaluation results
        """
        if not self.log_dir:
            return
        
        timestamp = datetime.now().isoformat()
        
        with open(self.summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for opponent, result in results.items():
                # Find corresponding match history entry
                match = next(
                    (m for m in reversed(self.match_history)
                     if m['baseline'] == opponent),
                    None
                )
                
                if match:
                    writer.writerow([
                        timestamp,
                        timestep,
                        opponent,
                        result['wins'],
                        result['losses'],
                        result['win_rate'],
                        match['elo_before'],
                        match['elo_after'],
                        match['elo_change']
                    ])
    
    def evaluate_against_baseline(
        self,
        agent_wins: int,
        agent_losses: int,
        baseline_name: str
    ) -> float:
        """Evaluate agent against baseline.
        
        Args:
            agent_wins: Number of wins
            agent_losses: Number of losses
            baseline_name: Name of baseline bot
            
        Returns:
            Updated Elo rating
        """
        baseline_elo = self.baseline_elos.get(baseline_name, self.INITIAL_ELO)
        
        total_matches = agent_wins + agent_losses
        if total_matches == 0:
            return self.agent_elo
        
        # Calculate expected score
        expected_score = self._expected_score(self.agent_elo, baseline_elo)
        
        # Calculate actual score
        actual_score = agent_wins / total_matches
        
        # Update Elo
        elo_change = self.K_FACTOR * (actual_score - expected_score)
        self.agent_elo += elo_change
        
        # Record match
        self.match_history.append({
            "baseline": baseline_name,
            "wins": agent_wins,
            "losses": agent_losses,
            "elo_before": self.agent_elo - elo_change,
            "elo_after": self.agent_elo,
            "elo_change": elo_change
        })
        
        return self.agent_elo
    
    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A vs player B.
        
        Args:
            elo_a: Elo rating of player A
            elo_b: Elo rating of player B
            
        Returns:
            Expected score (0-1)
        """
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    
    def get_elo(self) -> float:
        """Get current Elo rating.
        
        Returns:
            Current Elo rating
        """
        return self.agent_elo
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "current_elo": self.agent_elo,
            "matches_played": len(self.match_history),
            "match_history": self.match_history[-10:]  # Last 10 matches
        }
    
    def plot_elo_history(self, save_path: str):
        """Plot Elo rating history.
        
        Args:
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.match_history:
                return
            
            elos = [m["elo_after"] for m in self.match_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(elos)
            plt.xlabel("Match Number")
            plt.ylabel("Elo Rating")
            plt.title("Elo Rating Over Time")
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
