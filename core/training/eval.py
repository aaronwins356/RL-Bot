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


class EloRating:
    """Simple Elo rating calculator for game evaluation."""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        """Initialize Elo rating system.
        
        Args:
            k_factor: K-factor for rating updates (higher = more volatile)
            initial_rating: Initial rating for new players
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B.
        
        Args:
            rating_a: Elo rating of player A
            rating_b: Elo rating of player B
            
        Returns:
            Expected score (0-1)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    
    def update_rating(
        self,
        rating: float,
        opponent_rating: float,
        score: float
    ) -> float:
        """Update Elo rating based on game result.
        
        Args:
            rating: Current rating of player
            opponent_rating: Rating of opponent
            score: Game result (1.0 = win, 0.5 = draw, 0.0 = loss)
            
        Returns:
            Updated rating
        """
        expected = self.expected_score(rating, opponent_rating)
        return rating + self.k_factor * (score - expected)


def plot_elo_history(
    history: List[Dict[str, Any]],
    save_path: str,
    title: str = "Elo Rating Over Time"
):
    """Plot Elo rating history.
    
    Args:
        history: List of rating history entries with 'elo' key
        save_path: Path to save plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        if not history:
            logger.warning("No history to plot")
            return
        
        # Extract elo ratings
        elos = [entry.get('elo', entry.get('elo_after', 1500)) for entry in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(elos, linewidth=2)
        plt.xlabel("Game Number", fontsize=12)
        plt.ylabel("Elo Rating", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Elo plot saved to {save_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot")
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")


class EloEvaluator:
    """Evaluator for computing Elo ratings vs baselines."""
    
    # Elo calculation constants
    K_FACTOR = 32
    INITIAL_ELO = 1500
    
    def __init__(
        self,
        baseline_bots: Optional[List[str]] = None,
        log_dir: Optional[Path] = None,
        k_factor: float = 32
    ):
        """Initialize evaluator.
        
        Args:
            baseline_bots: List of baseline bot names
            log_dir: Directory for saving eval logs
            k_factor: K-factor for Elo rating updates
        """
        self.baseline_bots = baseline_bots or ["rookie", "allstar", "nexto"]
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Elo rating system
        self.elo_system = EloRating(k_factor=k_factor, initial_rating=self.INITIAL_ELO)
        
        # Elo ratings
        self.agent_elo = self.INITIAL_ELO
        self.baseline_elos = {
            "rookie": 800,
            "allstar": 1500,
            "nexto": 2000,
            "rule_policy": 1200,
            "baseline_ml": 1300,
            "previous_checkpoint": self.INITIAL_ELO
        }
        
        # Match history
        self.match_history = []
        
        # Enhanced metrics tracking
        self.curriculum_stage_elos = {}  # Elo per curriculum stage
        self.advanced_metrics = {
            'expected_value': [],
            'strategy_score': [],
            'action_entropy': [],
            'win_rate_vs_baseline': {},
            'checkpoint_comparisons': []
        }
        
        # Game-by-game CSV
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.summary_csv = self.log_dir / "eval_summary.csv"
            self.game_csv = self.log_dir / "game_by_game.csv"
            self.metrics_csv = self.log_dir / "advanced_metrics.csv"
            self._init_summary_csv()
            self._init_game_csv()
            self._init_metrics_csv()
    
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
    
    def _init_game_csv(self):
        """Initialize game-by-game CSV file."""
        if not self.game_csv.exists():
            with open(self.game_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'game_idx',
                    'opponent',
                    'result',
                    'our_score',
                    'opp_score',
                    'goal_diff',
                    'elo_before',
                    'elo_after',
                    'elo_change',
                    'expected_score'
                ])
    
    def _init_metrics_csv(self):
        """Initialize advanced metrics CSV file."""
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'timestep',
                    'curriculum_stage',
                    'elo',
                    'expected_value',
                    'strategy_score',
                    'action_entropy',
                    'win_rate_baseline',
                    'win_rate_checkpoint'
                ])
    
    def record_game(
        self,
        opponent: str,
        result: str,
        our_score: int = 0,
        opp_score: int = 0,
        game_idx: int = 0
    ):
        """Record a single game result.
        
        Args:
            opponent: Opponent identifier
            result: 'win', 'loss', or 'draw'
            our_score: Our goals scored
            opp_score: Opponent goals scored
            game_idx: Game index in series
        """
        if not self.log_dir:
            return
        
        opponent_elo = self.baseline_elos.get(opponent, self.INITIAL_ELO)
        elo_before = self.agent_elo
        expected_score = self.elo_system.expected_score(self.agent_elo, opponent_elo)
        
        # Determine score
        score = 1.0 if result == 'win' else (0.0 if result == 'loss' else 0.5)
        
        # Update Elo
        self.agent_elo = self.elo_system.update_rating(
            self.agent_elo, opponent_elo, score
        )
        elo_change = self.agent_elo - elo_before
        
        # Write to CSV
        timestamp = datetime.now().isoformat()
        with open(self.game_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                game_idx,
                opponent,
                result,
                our_score,
                opp_score,
                our_score - opp_score,
                elo_before,
                self.agent_elo,
                elo_change,
                expected_score
            ])
        
        # Add to match history
        self.match_history.append({
            'opponent': opponent,
            'result': result,
            'elo_before': elo_before,
            'elo_after': self.agent_elo,
            'elo_change': elo_change
        })
    
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
        expected_score = self.elo_system.expected_score(self.agent_elo, baseline_elo)
        
        # Calculate actual score
        actual_score = agent_wins / total_matches
        
        # Update Elo using the Elo system
        elo_before = self.agent_elo
        self.agent_elo = elo_before + self.elo_system.k_factor * (actual_score - expected_score)
        elo_change = self.agent_elo - elo_before
        
        # Record match
        self.match_history.append({
            "baseline": baseline_name,
            "wins": agent_wins,
            "losses": agent_losses,
            "elo_before": elo_before,
            "elo_after": self.agent_elo,
            "elo_change": elo_change
        })
        
        return self.agent_elo
    
    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A vs player B.
        
        DEPRECATED: Use elo_system.expected_score instead.
        
        Args:
            elo_a: Elo rating of player A
            elo_b: Elo rating of player B
            
        Returns:
            Expected score (0-1)
        """
        return self.elo_system.expected_score(elo_a, elo_b)
    
    def record_curriculum_stage_elo(self, stage: int, elo: float):
        """Record Elo for specific curriculum stage.
        
        Args:
            stage: Curriculum stage index
            elo: Elo rating at this stage
        """
        if stage not in self.curriculum_stage_elos:
            self.curriculum_stage_elos[stage] = []
        self.curriculum_stage_elos[stage].append(elo)
    
    def record_advanced_metrics(
        self,
        timestep: int,
        curriculum_stage: int,
        model,
        observations: np.ndarray
    ):
        """Record advanced evaluation metrics.
        
        Args:
            timestep: Current timestep
            curriculum_stage: Current curriculum stage
            model: Model being evaluated
            observations: Sample observations for metrics
        """
        import torch
        
        # Compute expected value of state
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations)
            if hasattr(model, 'get_value'):
                values = model.get_value(obs_tensor)
                expected_value = float(values.mean().item())
            else:
                expected_value = 0.0
        
        # Compute action entropy (strategy diversity)
        with torch.no_grad():
            if hasattr(model, 'forward'):
                action_dist, _ = model(obs_tensor)
                if hasattr(action_dist, 'entropy'):
                    entropy = float(action_dist.entropy().mean().item())
                else:
                    entropy = 0.0
            else:
                entropy = 0.0
        
        # Store metrics
        self.advanced_metrics['expected_value'].append(expected_value)
        self.advanced_metrics['action_entropy'].append(entropy)
        
        # Compute strategy score (placeholder - would be based on specific strategy analysis)
        strategy_score = expected_value * 0.5 + entropy * 0.5
        self.advanced_metrics['strategy_score'].append(strategy_score)
        
        # Record to CSV
        if self.log_dir:
            with open(self.metrics_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    timestep,
                    curriculum_stage,
                    self.agent_elo,
                    expected_value,
                    strategy_score,
                    entropy,
                    self.advanced_metrics['win_rate_vs_baseline'].get('latest', 0.0),
                    0.0  # Placeholder for checkpoint win rate
                ])
    
    def compare_checkpoints(
        self,
        checkpoint1_path: str,
        checkpoint2_path: str,
        num_games: int = 10
    ) -> Dict[str, Any]:
        """Compare two checkpoints head-to-head.
        
        Args:
            checkpoint1_path: Path to first checkpoint
            checkpoint2_path: Path to second checkpoint
            num_games: Number of games to play
            
        Returns:
            Comparison results
        """
        # Placeholder - would need actual game playing
        # In real implementation, would load both checkpoints and play matches
        
        comparison = {
            'checkpoint1': checkpoint1_path,
            'checkpoint2': checkpoint2_path,
            'games_played': num_games,
            'checkpoint1_wins': 0,
            'checkpoint2_wins': 0,
            'draws': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Simulate results
        import random
        for _ in range(num_games):
            result = random.choice(['c1', 'c2', 'draw'])
            if result == 'c1':
                comparison['checkpoint1_wins'] += 1
            elif result == 'c2':
                comparison['checkpoint2_wins'] += 1
            else:
                comparison['draws'] += 1
        
        self.advanced_metrics['checkpoint_comparisons'].append(comparison)
        
        logger.info(
            f"Checkpoint comparison: {comparison['checkpoint1_wins']}-"
            f"{comparison['checkpoint2_wins']}-{comparison['draws']}"
        )
        
        return comparison
    
    def plot_advanced_metrics(self, save_dir: Optional[str] = None):
        """Plot advanced evaluation metrics over time.
        
        Args:
            save_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            
            save_dir = Path(save_dir) if save_dir else self.log_dir
            if not save_dir:
                logger.warning("No save directory specified for plots")
                return
            
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot expected value over time
            if self.advanced_metrics['expected_value']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.advanced_metrics['expected_value'])
                plt.xlabel('Evaluation Step')
                plt.ylabel('Expected Value')
                plt.title('Expected Value of States Over Time')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'expected_value.png', dpi=150)
                plt.close()
            
            # Plot action entropy
            if self.advanced_metrics['action_entropy']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.advanced_metrics['action_entropy'])
                plt.xlabel('Evaluation Step')
                plt.ylabel('Action Entropy')
                plt.title('Action Entropy (Strategy Diversity) Over Time')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'action_entropy.png', dpi=150)
                plt.close()
            
            # Plot strategy score
            if self.advanced_metrics['strategy_score']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.advanced_metrics['strategy_score'])
                plt.xlabel('Evaluation Step')
                plt.ylabel('Strategy Score')
                plt.title('Overall Strategy Score Over Time')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'strategy_score.png', dpi=150)
                plt.close()
            
            # Plot curriculum stage Elos
            if self.curriculum_stage_elos:
                plt.figure(figsize=(12, 6))
                for stage, elos in self.curriculum_stage_elos.items():
                    plt.plot(elos, label=f'Stage {stage}')
                plt.xlabel('Evaluation')
                plt.ylabel('Elo Rating')
                plt.title('Elo Rating by Curriculum Stage')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'curriculum_elos.png', dpi=150)
                plt.close()
            
            logger.info(f"Advanced metric plots saved to {save_dir}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping advanced plots")
        except Exception as e:
            logger.error(f"Failed to generate advanced plots: {e}")
    
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
        plot_elo_history(self.match_history, save_path)
