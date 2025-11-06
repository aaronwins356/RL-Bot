"""Training loop for RL bot.

This module implements the main training loop with PPO.
"""
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from core.models.nets import ActorCriticNet
from core.models.ppo import PPO
from core.training.buffer import ReplayBuffer
from core.training.selfplay import SelfPlayManager
from core.training.eval import EloEvaluator
from core.infra.config import Config
from core.infra.logging import MetricsLogger
from core.infra.checkpoints import CheckpointManager


class TrainingLoop:
    """Main training loop for PPO."""
    
    def __init__(
        self,
        config: Config,
        log_dir: Optional[Path] = None
    ):
        """Initialize training loop.
        
        Args:
            config: Training configuration
            log_dir: Log directory
        """
        self.config = config
        self.log_dir = Path(log_dir or config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(config.device)
        
        # Model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Algorithm
        self.ppo = PPO(self.model, config.raw_config.get("training", {}))
        
        # Buffer
        self.buffer = ReplayBuffer(
            capacity=config.raw_config.get("telemetry", {}).get("buffer_size", 100000)
        )
        
        # Managers
        self.logger = MetricsLogger(
            self.log_dir,
            use_tensorboard=config.tensorboard
        )
        self.checkpoint_manager = CheckpointManager(
            Path(config.save_dir),
            keep_best_n=config.raw_config.get("checkpoints", {}).get("keep_best_n", 5)
        )
        self.selfplay_manager = SelfPlayManager(
            config.raw_config.get("training", {}).get("selfplay", {})
        )
        self.evaluator = EloEvaluator()
        
        # Training state
        self.timestep = 0
        self.episode = 0
    
    def _create_model(self) -> ActorCriticNet:
        """Create model from config.
        
        Returns:
            ActorCritic network
        """
        network_config = self.config.raw_config.get("network", {})
        
        # TODO: Get actual observation size from encoder
        obs_size = 173  # Placeholder
        
        model = ActorCriticNet(
            input_size=obs_size,
            hidden_sizes=self.config.hidden_sizes,
            action_categoricals=5,
            action_bernoullis=3,
            activation=self.config.activation,
            use_lstm=self.config.use_lstm
        )
        
        return model
    
    def train(self, total_timesteps: Optional[int] = None):
        """Run training loop.
        
        Args:
            total_timesteps: Total timesteps to train (uses config if None)
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        print(f"Starting training for {total_timesteps} timesteps")
        print(f"Device: {self.device}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        
        while self.timestep < total_timesteps:
            # Collect experience (would need environment)
            # For now, this is a placeholder
            # In real implementation, this would interact with RLBot/RLGym
            
            # Update model if buffer has enough data
            if len(self.buffer) >= self.config.batch_size:
                self._update()
            
            # Log
            if self.timestep % self.config.log_interval == 0:
                self._log_progress()
            
            # Save checkpoint
            if self.timestep % self.config.save_interval == 0:
                self._save_checkpoint()
            
            # Evaluate
            eval_interval = self.config.raw_config.get("logging", {}).get("eval_interval", 50000)
            if self.timestep % eval_interval == 0:
                self._evaluate()
            
            self.timestep += 1
        
        print("Training complete!")
        self._save_checkpoint(is_final=True)
    
    def _update(self):
        """Perform PPO update."""
        # Get trajectory from buffer
        trajectory = self.buffer.get_recent_trajectory(max_length=self.config.batch_size)
        
        if len(trajectory["observations"]) == 0:
            return
        
        # Convert to tensors
        obs = torch.tensor(trajectory["observations"], dtype=torch.float32, device=self.device)
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]
        
        # Compute values
        with torch.no_grad():
            values = self.model.get_value(obs).cpu().numpy().squeeze()
        
        # Compute advantages and returns
        next_value = values[-1] if len(values) > 0 else 0.0
        advantages, returns = self.ppo.compute_gae(
            rewards, values[:-1] if len(values) > 1 else np.array([]), dones, next_value
        )
        
        # For now, skip actual PPO update (would need proper action log probs)
        # In full implementation, this would call ppo.update()
    
    def _log_progress(self):
        """Log training progress."""
        buffer_stats = self.buffer.get_stats()
        ppo_stats = self.ppo.get_stats()
        selfplay_stats = self.selfplay_manager.get_stats()
        
        self.logger.log_dict({
            "timestep": self.timestep,
            "episode": self.episode,
            "buffer_size": buffer_stats["size"],
            "avg_reward": buffer_stats.get("avg_reward_per_episode", 0.0)
        })
        
        self.logger.flush()
        
        print(f"Timestep: {self.timestep}, Episode: {self.episode}")
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save checkpoint."""
        metrics = {
            "timestep": self.timestep,
            "episode": self.episode,
            "eval_score": self.evaluator.get_elo()
        }
        
        self.checkpoint_manager.save_checkpoint(
            self.model,
            self.ppo.optimizer,
            self.timestep,
            metrics,
            is_best=is_final
        )
    
    def _evaluate(self):
        """Evaluate model."""
        # Placeholder for evaluation
        # In full implementation, this would play matches against baselines
        elo = self.evaluator.get_elo()
        
        self.logger.log_scalar("eval/elo", elo, self.timestep)
        self.logger.flush()
        
        print(f"Evaluation - Elo: {elo:.0f}")
