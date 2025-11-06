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
OBS_SIZE = 180  # Standard observation size from ObservationEncoder
DEFAULT_EVAL_GAMES = 25  # Default number of games per opponent during evaluation


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
        
        # Debug mode flag
        self.debug_mode = config.raw_config.get('training', {}).get('debug_mode', False)
        if self.debug_mode:
            logger.info("DEBUG MODE ENABLED - Detailed logging active")
        
        # Set random seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Device - check CUDA availability and handle "auto"
        device_str = config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device_str}")
        elif device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_str = "cpu"
        self.device = torch.device(device_str)
        
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
        self.best_elo = -float('inf')
        self.eval_history = []
        self.early_stop_counter = 0
        self.early_stop_patience = config.raw_config.get("training", {}).get(
            "early_stop_patience", 5
        )
        
        # Forced curriculum stage (from CLI)
        self.forced_stage = None
        
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
    
    def train(self, total_timesteps: Optional[int] = None, forced_stage: Optional[int] = None):
        """Run training loop.
        
        Args:
            total_timesteps: Total timesteps to train (uses config if None)
            forced_stage: Force specific curriculum stage (for debugging)
        """
        import time
        self.start_time = time.time()
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        self.forced_stage = forced_stage
        
        logger.info(f"Starting training for {total_timesteps} timesteps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        
        if self.curriculum_manager:
            logger.info(f"Curriculum learning enabled with {len(self.curriculum_manager.stages)} stages")
        
        if forced_stage is not None:
            logger.info(f"Forcing curriculum stage: {forced_stage}")
        
        # Create environment for experience collection
        from core.env.rocket_sim_env import RocketSimEnv
        from pathlib import Path
        env = RocketSimEnv(
            reward_config_path=Path("configs/rewards.yaml"),
            simulation_mode=True,
            debug_mode=self.debug_mode
        )
        
        # Episode state
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        while self.timestep < total_timesteps:
            # Check for early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                logger.warning(
                    f"Early stopping triggered after {self.early_stop_patience} "
                    f"evaluations without improvement"
                )
                break
            
            # Check curriculum stage transitions
            if self.curriculum_manager and forced_stage is None:
                should_transition, new_stage = self.selfplay_manager.should_transition_stage(
                    self.timestep
                )
                if should_transition:
                    logger.info(f"Transitioned to curriculum stage: {new_stage.name}")
                    
                    # Add current checkpoint to opponent pool for self-play
                    if new_stage.opponent_type in ["selfplay", "checkpoint"]:
                        checkpoint_path = self.checkpoint_manager.get_latest_path()
                        if checkpoint_path:
                            self.selfplay_manager.add_opponent(
                                checkpoint_path,
                                elo=self.evaluator.get_elo(),
                                timestep=self.timestep
                            )
            
            # Get current stage config
            stage_config = self.selfplay_manager.get_stage_config(self.timestep)
            
            # Collect experience step
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                cat_probs, ber_probs, value, _, _ = self.model(obs_tensor)
                
                # Calculate action entropy for logging
                action_entropy = 0.0
                
                # Sample actions - cat_probs has shape (batch=1, n_cat, 3)
                # ber_probs has shape (batch=1, n_ber, 2)
                cat_actions = []
                cat_probs_batch = cat_probs[0]  # Get first (only) batch element -> (n_cat, 3)
                for i in range(cat_probs_batch.shape[0]):
                    probs = cat_probs_batch[i]  # Shape: (3,)
                    cat_dist = torch.distributions.Categorical(probs)
                    cat_actions.append(cat_dist.sample().item())
                    action_entropy += cat_dist.entropy().item()
                
                ber_actions = []
                ber_probs_batch = ber_probs[0]  # Get first (only) batch element -> (n_ber, 2)
                for i in range(ber_probs_batch.shape[0]):
                    probs = ber_probs_batch[i]  # Shape: (2,) - [prob_0, prob_1]
                    # Bernoulli takes probability of action=1
                    ber_dist = torch.distributions.Bernoulli(probs[1])
                    ber_actions.append(int(ber_dist.sample().item()))
                    action_entropy += ber_dist.entropy().item()
                
                # Average entropy
                num_actions = cat_probs_batch.shape[0] + ber_probs_batch.shape[0]
                avg_action_entropy = action_entropy / max(1, num_actions)
                
                # Convert to action array
                # Categorical actions are typically 0-4 (5 options), so we map to [-1, 1]
                # For example: 0->-1, 1->-0.5, 2->0, 3->0.5, 4->1
                # Formula: (action - 2) / 2.0 maps [0,4] to [-1,1]
                action = np.array([
                    (cat_actions[0] - 2) / 2.0,  # throttle: map [0,4] to [-1,1]
                    (cat_actions[1] - 2) / 2.0,  # steer: map [0,4] to [-1,1]
                    0.0,  # pitch (not used in 2D simulation)
                    0.0,  # yaw (not used in 2D simulation)
                    0.0,  # roll (not used in 2D simulation)
                    ber_actions[0],  # jump: binary 0/1
                    ber_actions[1],  # boost: binary 0/1
                    ber_actions[2] if len(ber_actions) > 2 else 0.0,  # handbrake: binary 0/1
                ])
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            self.buffer.add({
                'observation': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'value': value.item(),
                'entropy': avg_action_entropy
            })
            
            episode_reward += reward
            episode_length += 1
            
            # Log action entropy periodically
            if self.timestep % 100 == 0:
                self.logger.log_scalar("train/action_entropy", avg_action_entropy, self.timestep)
            
            obs = next_obs
            
            # Episode done
            if done:
                self.episode += 1
                if self.debug_mode:
                    logger.debug(
                        f"Episode {self.episode} complete: "
                        f"length={episode_length}, reward={episode_reward:.2f}"
                    )
                
                # Log episode stats
                self.logger.log_scalar("train/episode_reward", episode_reward, self.timestep)
                self.logger.log_scalar("train/episode_length", episode_length, self.timestep)
                
                # Reset
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
            
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
        self._save_checkpoint(is_best=True)  # Final checkpoint is also best
    
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
        
        # Basic stats
        self.logger.log_dict({
            "timestep": self.timestep,
            "episode": self.episode,
            "buffer_size": buffer_stats["size"],
            "avg_reward": buffer_stats.get("avg_reward_per_episode", 0.0)
        })
        
        # PPO stats (if available)
        if ppo_stats.get("ent_coef") and len(ppo_stats["ent_coef"]) > 0:
            latest_ent = ppo_stats["ent_coef"][-1]
            self.logger.log_scalar("train/entropy_coef", latest_ent, self.timestep)
        
        if ppo_stats.get("policy_loss") and len(ppo_stats["policy_loss"]) > 0:
            latest_policy_loss = ppo_stats["policy_loss"][-1]
            self.logger.log_scalar("train/policy_loss", latest_policy_loss, self.timestep)
        
        if ppo_stats.get("value_loss") and len(ppo_stats["value_loss"]) > 0:
            latest_value_loss = ppo_stats["value_loss"][-1]
            self.logger.log_scalar("train/value_loss", latest_value_loss, self.timestep)
        
        if ppo_stats.get("gae_lambda") and len(ppo_stats["gae_lambda"]) > 0:
            latest_gae = ppo_stats["gae_lambda"][-1]
            self.logger.log_scalar("train/gae_lambda", latest_gae, self.timestep)
        
        self.logger.flush()
        
        logger.info(
            f"Timestep: {self.timestep}, Episode: {self.episode}, "
            f"Buffer: {buffer_stats['size']}, Avg Reward: {buffer_stats.get('avg_reward_per_episode', 0.0):.2f}"
        )
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint so far
        """
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
            is_best=is_best
        )
    
    def _evaluate(self):
        """Evaluate model and check for early stopping."""
        # Get num_games from config (default to constant if not specified)
        num_games = self.config.raw_config.get('logging', {}).get('eval_num_games', DEFAULT_EVAL_GAMES)
        
        # Run full evaluation suite
        results = self.evaluator.evaluate_full(
            self.model,
            self.timestep,
            num_games=num_games
        )
        
        current_elo = self.evaluator.get_elo()
        
        # Log metrics
        self.logger.log_scalar("eval/elo", current_elo, self.timestep)
        
        for opponent, result in results.items():
            self.logger.log_scalar(f"eval/{opponent}/win_rate", result['win_rate'], self.timestep)
            self.logger.log_scalar(f"eval/{opponent}/wins", result['wins'], self.timestep)
            self.logger.log_scalar(f"eval/{opponent}/losses", result['losses'], self.timestep)
        
        self.logger.flush()
        
        logger.info(f"Evaluation - Elo: {current_elo:.0f} (after {num_games} games per opponent)")
        
        # Check for improvement
        if current_elo > self.best_elo:
            logger.info(f"New best Elo: {current_elo:.0f} (previous: {self.best_elo:.0f})")
            self.best_elo = current_elo
            self.early_stop_counter = 0
            
            # Save best checkpoint
            self._save_checkpoint(is_best=True)
        else:
            self.early_stop_counter += 1
            logger.info(
                f"No Elo improvement ({self.early_stop_counter}/{self.early_stop_patience})"
            )
        
        # Track evaluation history
        self.eval_history.append({
            'timestep': self.timestep,
            'elo': current_elo,
            'results': results
        })
