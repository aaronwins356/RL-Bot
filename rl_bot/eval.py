"""
Evaluation script for measuring bot performance.
Includes Elo rating system and performance metrics.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import json

from rl_bot.core.model import PPOAgent


class EloRating:
    """
    Elo rating system for tracking bot performance.
    """
    
    def __init__(self, initial_rating: float = 1000, k_factor: float = 32):
        """
        Args:
            initial_rating: Initial Elo rating
            k_factor: K-factor for rating updates (higher = more volatile)
        """
        self.rating = initial_rating
        self.k_factor = k_factor
        self.history = [initial_rating]
    
    def expected_score(self, opponent_rating: float) -> float:
        """
        Calculate expected score against opponent.
        
        Args:
            opponent_rating: Opponent's Elo rating
            
        Returns:
            Expected score (0 to 1)
        """
        return 1.0 / (1.0 + 10 ** ((opponent_rating - self.rating) / 400))
    
    def update(self, opponent_rating: float, actual_score: float):
        """
        Update Elo rating based on match result.
        
        Args:
            opponent_rating: Opponent's Elo rating
            actual_score: Actual score (1 for win, 0.5 for draw, 0 for loss)
        """
        expected = self.expected_score(opponent_rating)
        self.rating += self.k_factor * (actual_score - expected)
        self.history.append(self.rating)
    
    def get_rating(self) -> float:
        """Get current Elo rating."""
        return self.rating
    
    def get_history(self) -> List[float]:
        """Get rating history."""
        return self.history


class Evaluator:
    """
    Evaluates bot performance using multiple metrics.
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        env,
        config: Dict[str, Any],
        logger=None
    ):
        """
        Args:
            agent: PPO agent to evaluate
            env: Evaluation environment
            config: Configuration dictionary
            logger: Logger instance
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger
        
        eval_config = config.get('evaluation', {})
        self.num_episodes = eval_config.get('num_eval_episodes', 20)
        self.deterministic = eval_config.get('eval_deterministic', True)
        self.track_elo = eval_config.get('track_elo', True)
        
        # Elo rating
        if self.track_elo:
            self.elo = EloRating(
                initial_rating=eval_config.get('initial_elo', 1000),
                k_factor=eval_config.get('k_factor', 32)
            )
        else:
            self.elo = None
    
    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate agent over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate (uses config default if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
        
        episode_rewards = []
        episode_lengths = []
        wins = 0
        losses = 0
        draws = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Handle vectorized env (take first environment)
            if hasattr(self.env, 'num_envs'):
                obs = obs[0] if len(obs.shape) > 1 else obs
            
            while not done:
                # Get action
                action, _ = self.agent.predict(obs, deterministic=self.deterministic)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Handle vectorized env
                if hasattr(self.env, 'num_envs'):
                    if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                        obs = obs[0]
                    if isinstance(reward, (list, np.ndarray)):
                        reward = reward[0]
                    if isinstance(terminated, (list, np.ndarray)):
                        terminated = terminated[0]
                    if isinstance(truncated, (list, np.ndarray)):
                        truncated = truncated[0]
                    done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Determine win/loss/draw based on final reward
            if episode_reward > 5:  # Won
                wins += 1
                if self.elo:
                    self.elo.update(opponent_rating=1000, actual_score=1.0)
            elif episode_reward < -5:  # Lost
                losses += 1
                if self.elo:
                    self.elo.update(opponent_rating=1000, actual_score=0.0)
            else:  # Draw
                draws += 1
                if self.elo:
                    self.elo.update(opponent_rating=1000, actual_score=0.5)
        
        # Compute metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_episodes,
            'num_episodes': num_episodes
        }
        
        if self.elo:
            metrics['elo_rating'] = self.elo.get_rating()
        
        if self.logger:
            self.logger.info(f"Evaluation complete:")
            self.logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            self.logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            if self.elo:
                self.logger.info(f"  Elo Rating: {metrics['elo_rating']:.0f}")
        
        return metrics
    
    def plot_elo_history(self, save_path: Optional[str] = None):
        """
        Plot Elo rating history.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        if not self.elo:
            print("Elo tracking is disabled")
            return
        
        history = self.elo.get_history()
        
        plt.figure(figsize=(10, 6))
        plt.plot(history, linewidth=2)
        plt.xlabel('Evaluation Episode', fontsize=12)
        plt.ylabel('Elo Rating', fontsize=12)
        plt.title('Bot Elo Rating Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Elo plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def evaluate_checkpoint(
    checkpoint_path: str,
    env,
    config: Dict[str, Any],
    num_episodes: int = 20,
    plot_elo: bool = True,
    save_dir: Optional[str] = None,
    logger=None
) -> Dict[str, Any]:
    """
    Evaluate a saved checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        env: Evaluation environment
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes
        plot_elo: Whether to plot Elo history
        save_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Evaluation metrics
    """
    # Load model
    from rl_bot.core.utils import get_device, load_checkpoint
    
    device = get_device(config.get('device', 'auto'))
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    network_config = config.get('network', {})
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=network_config.get('hidden_sizes', [512, 512, 256]),
        activation=network_config.get('activation', 'relu'),
        device=device
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, agent.model, device=device)
    
    if logger:
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Checkpoint step: {checkpoint.get('global_step', 'unknown')}")
    
    # Create evaluator
    evaluator = Evaluator(agent, env, config, logger)
    
    # Run evaluation
    metrics = evaluator.evaluate(num_episodes)
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = save_dir / "eval_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if logger:
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Plot Elo history
        if plot_elo and evaluator.elo:
            elo_plot_path = save_dir / "elo_history.png"
            evaluator.plot_elo_history(str(elo_plot_path))
    
    return metrics


def compare_checkpoints(
    checkpoint_paths: List[str],
    env,
    config: Dict[str, Any],
    num_episodes: int = 20,
    save_dir: Optional[str] = None,
    logger=None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        env: Evaluation environment
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes per checkpoint
        save_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Dictionary mapping checkpoint name to metrics
    """
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = Path(checkpoint_path).stem
        
        if logger:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {checkpoint_name}")
            logger.info(f"{'='*50}")
        
        metrics = evaluate_checkpoint(
            checkpoint_path,
            env,
            config,
            num_episodes,
            plot_elo=False,
            save_dir=None,
            logger=logger
        )
        
        results[checkpoint_name] = metrics
    
    # Save comparison
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = save_dir / "checkpoint_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if logger:
            logger.info(f"\nComparison saved to {comparison_path}")
    
    # Print summary
    if logger:
        logger.info(f"\n{'='*50}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*50}")
        
        for name, metrics in results.items():
            logger.info(f"{name}:")
            logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            if 'elo_rating' in metrics:
                logger.info(f"  Elo: {metrics['elo_rating']:.0f}")
    
    return results
