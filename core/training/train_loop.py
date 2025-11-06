"""Training loop for RL bot.

This module implements the main training loop with PPO.
"""
import torch
import numpy as np
import logging
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

logger = logging.getLogger(__name__)

# Constants
OBS_SIZE = 173  # Standard observation size (placeholder - should come from encoder)


class TrainingLoop:
    """Main training loop for PPO."""
    
    def __init__(
        self,
        config: Config,
        log_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize training loop.
        
        Args:
            config: Training configuration
            log_dir: Log directory
            checkpoint_path: Path to checkpoint to resume from
            seed: Random seed for reproducibility
        """
        self.config = config
        self.log_dir = Path(log_dir or config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        
        # Set random seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
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
        self.evaluator = EloEvaluator(log_dir=self.log_dir)
        
        # Curriculum manager (if enabled)
        curriculum_config = config.raw_config.get("training", {}).get("curriculum", {})
        self.curriculum_manager = None
        if curriculum_config.get('aerial_focus', False) or curriculum_config:
            from core.training.curriculum import CurriculumManager
            self.curriculum_manager = CurriculumManager(curriculum_config)
        
        # Training state
        self.timestep = 0
        self.episode = 0
        
        # Offline pretraining (if enabled)
        if config.raw_config.get("training", {}).get("offline", {}).get("enabled", False):
            self._run_offline_pretraining()
    
    def _run_offline_pretraining(self):
        """Run offline pretraining with behavioral cloning."""
        offline_config = self.config.raw_config.get("training", {}).get("offline", {})
        dataset_path = Path(offline_config.get("dataset_path", "data/telemetry_logs"))
        pretrain_epochs = offline_config.get("pretrain_epochs", 10)
        pretrain_lr = offline_config.get("pretrain_lr", 1e-3)
        
        if not dataset_path.exists():
            logger.warning(f"Offline dataset not found at {dataset_path}, skipping pretraining")
            return
        
        logger.info(f"Starting offline pretraining for {pretrain_epochs} epochs...")
        
        try:
            from core.training.offline_dataset import OfflineDataset
            import torch.nn.functional as F
            
            # Load dataset
            dataset = OfflineDataset(dataset_path, max_samples=100000)
            dataloader = dataset.get_loader(batch_size=256, shuffle=True)
            
            # Setup optimizer for pretraining
            pretrain_optimizer = torch.optim.Adam(self.model.parameters(), lr=pretrain_lr)
            
            # Pretrain
            self.model.train()
            for epoch in range(pretrain_epochs):
                total_loss = 0.0
                num_batches = 0
                
                for batch in dataloader:
                    obs = torch.tensor(batch['observation'], dtype=torch.float32, device=self.device)
                    action = torch.tensor(batch['action'], dtype=torch.float32, device=self.device)
                    
                    # Forward pass (behavioral cloning)
                    # Note: This is simplified - real implementation would need proper action heads
                    value = self.model.get_value(obs)
                    
                    # Compute loss (placeholder - would need actual action prediction)
                    loss = F.mse_loss(value, torch.zeros_like(value))  # Placeholder
                    
                    # Backward pass
                    pretrain_optimizer.zero_grad()
                    loss.backward()
                    pretrain_optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                logger.info(f"  Epoch {epoch+1}/{pretrain_epochs}, Loss: {avg_loss:.4f}")
            
            logger.info("Offline pretraining completed!")
            
        except Exception as e:
            logger.warning(f"Offline pretraining failed: {e}")
    
    def _create_model(self) -> ActorCriticNet:
        """Create model from config.
        
        Returns:
            ActorCritic network
        """
        network_config = self.config.raw_config.get("network", {})
        
        model = ActorCriticNet(
            input_size=OBS_SIZE,
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
        
        logger.info(f"Starting training for {total_timesteps} timesteps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        
        if self.curriculum_manager:
            logger.info(f"Curriculum learning enabled with {len(self.curriculum_manager.stages)} stages")
        
        while self.timestep < total_timesteps:
            # Check curriculum stage transitions
            if self.curriculum_manager:
                if self.curriculum_manager.should_transition(self.timestep):
                    stage = self.curriculum_manager.get_current_stage(self.timestep)
                    logger.info(f"Transitioned to curriculum stage: {stage.name}")
            
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
        
        logger.info("Training complete!")
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
        
        logger.info(f"Timestep: {self.timestep}, Episode: {self.episode}")
    
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
        # Run full evaluation suite
        results = self.evaluator.evaluate_full(
            self.model,
            self.timestep,
            num_games=5
        )
        
        # Log metrics
        self.logger.log_scalar("eval/elo", self.evaluator.get_elo(), self.timestep)
        
        for opponent, result in results.items():
            self.logger.log_scalar(f"eval/{opponent}/win_rate", result['win_rate'], self.timestep)
            self.logger.log_scalar(f"eval/{opponent}/wins", result['wins'], self.timestep)
            self.logger.log_scalar(f"eval/{opponent}/losses", result['losses'], self.timestep)
        
        self.logger.flush()
        
        logger.info(f"Evaluation - Elo: {self.evaluator.get_elo():.0f}")
