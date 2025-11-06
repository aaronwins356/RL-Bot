"""Evaluation module for computing Elo ratings and generating plots.

This module provides evaluation against scripted baselines.
"""
from typing import Dict, Any, List, Optional
import numpy as np


class EloEvaluator:
    """Evaluator for computing Elo ratings vs baselines."""
    
    # Elo calculation constants
    K_FACTOR = 32
    INITIAL_ELO = 1500
    
    def __init__(self, baseline_bots: Optional[List[str]] = None):
        """Initialize evaluator.
        
        Args:
            baseline_bots: List of baseline bot names
        """
        self.baseline_bots = baseline_bots or ["rookie", "allstar", "nexto"]
        
        # Elo ratings
        self.agent_elo = self.INITIAL_ELO
        self.baseline_elos = {
            "rookie": 800,
            "allstar": 1500,
            "nexto": 2000
        }
        
        # Match history
        self.match_history = []
    
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
            print("Matplotlib not available, skipping plot")
