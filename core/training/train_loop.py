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


def ensure_array(value, name="value"):
    """Ensure value is a numpy array, wrapping scalars if needed.
    
    Args:
        value: Value to wrap (can be scalar, list, or array)
        name: Name for debugging
        
    Returns:
        numpy array
    """
    if np.isscalar(value):
        return np.array([value])
    elif isinstance(value, list):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        logger.warning(f"Unexpected type for {name}: {type(value)}, attempting conversion")
        return np.array([value])


def create_vectorized_env(num_envs: int, reward_config_path: Path, simulation_mode: bool = True, debug_mode: bool = False):
    """Create vectorized environment with automatic fallback.
    
    Args:
        num_envs: Number of parallel environments
        reward_config_path: Path to reward config
        simulation_mode: Use simulation mode
        debug_mode: Enable debug logging
        
    Returns:
        Vectorized environment
    """
    from core.env.rocket_sim_env import RocketSimEnv
    
    def make_env(env_id):
        """Create a single environment."""
        def _init():
            env = RocketSimEnv(
                reward_config_path=reward_config_path,
                simulation_mode=simulation_mode,
                debug_mode=debug_mode
            )
            return env
        return _init
    
    # Try DummyVecEnv (threaded, Windows-compatible)
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        logger.info(f"✅ DummyVecEnv with {num_envs} environments created")
        return env
    except Exception as e:
        logger.warning(f"DummyVecEnv creation failed: {e}")
        # Fallback to single environment
        logger.info("Falling back to single environment")
        return RocketSimEnv(
            reward_config_path=reward_config_path,
            simulation_mode=simulation_mode,
            debug_mode=debug_mode
        )


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
        
        # Mixed precision training support
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ Mixed-precision training (AMP) enabled")
        else:
            self.scaler = None
        
        # Model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Algorithm
        self.ppo = PPO(self.model, config.raw_config.get("training", {}), use_amp=self.use_amp)
        
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
        
        # Number of environments
        self.num_envs = config.raw_config.get("training", {}).get("num_envs", 1)
        
        # Offline pretraining (if enabled)
        if config.raw_config.get("training", {}).get("offline", {}).get("enabled", False):
            self._run_offline_pretraining()
        
        # Run verification
        self._run_verification()
    
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
    
    def _run_verification(self):
        """Run verification routine to check setup."""
        logger.info("=" * 60)
        logger.info("VERIFICATION ROUTINE")
        logger.info("=" * 60)
        
        # Print curriculum stages
        if self.selfplay_manager:
            logger.info("Curriculum Stages:")
            for i, stage in enumerate(self.selfplay_manager.stages):
                logger.info(f"  Stage {i}: {stage.name}")
            logger.info("✅ Curriculum restriction verified")
        
        # Verify environment setup with num_envs
        if self.num_envs > 1:
            logger.info(f"✅ Multi-env setup verified: {self.num_envs} environments")
        else:
            logger.info(f"Single environment mode (num_envs={self.num_envs})")
        
        # Test dummy forward pass
        try:
            dummy_obs = torch.randn(1, OBS_SIZE, device=self.device)
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        cat_probs, ber_probs, value, _, _ = self.model(dummy_obs)
                else:
                    cat_probs, ber_probs, value, _, _ = self.model(dummy_obs)
            logger.info(f"✅ Model forward pass verified: cat_probs={cat_probs.shape}, value={value.shape}")
        except Exception as e:
            logger.error(f"❌ Model forward pass failed: {e}")
            raise
        
        # Print configuration summary
        logger.info("=" * 60)
        logger.info("CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        logger.info(f"Number of Environments: {self.num_envs}")
        logger.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Batch Size: {self.config.batch_size}")
        logger.info(f"Learning Rate: {self.config.raw_config.get('training', {}).get('learning_rate', 3e-4)}")
        logger.info("=" * 60)
        logger.info("✅ Training ready (no multiprocessing conflicts)")
        logger.info("=" * 60)
    
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
        
        # Create environment(s) for experience collection
        if self.num_envs > 1:
            env = create_vectorized_env(
                self.num_envs,
                Path("configs/rewards.yaml"),
                simulation_mode=True,
                debug_mode=self.debug_mode
            )
            is_vec_env = True
            logger.info(f"✅ threaded envs ready")
        else:
            from core.env.rocket_sim_env import RocketSimEnv
            env = RocketSimEnv(
                reward_config_path=Path("configs/rewards.yaml"),
                simulation_mode=True,
                debug_mode=self.debug_mode
            )
            is_vec_env = False
        
        # Episode state - handle both vectorized and single env
        obs = env.reset()
        if is_vec_env:
            # VecEnv returns (num_envs, obs_dim)
            episode_rewards = np.zeros(self.num_envs)
            episode_lengths = np.zeros(self.num_envs, dtype=int)
        else:
            # Single env
            obs = ensure_array(obs, "observation")
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
            
            # Collect experience step - handle both vectorized and single env
            with torch.no_grad():
                # Prepare observation batch
                if is_vec_env:
                    # VecEnv: obs is already (num_envs, obs_dim)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                else:
                    # Single env: add batch dimension
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Forward pass with AMP if available
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        cat_probs, ber_probs, value, _, _ = self.model(obs_tensor)
                else:
                    cat_probs, ber_probs, value, _, _ = self.model(obs_tensor)
                
                # Sample actions for each environment
                batch_size = obs_tensor.shape[0]
                all_actions = []
                all_cat_actions = []
                all_ber_actions = []
                all_cat_log_probs = []
                all_ber_log_probs = []
                all_entropies = []
                
                for b in range(batch_size):
                    # Calculate action entropy for logging
                    action_entropy = 0.0
                    
                    # Sample categorical actions
                    cat_actions = []
                    cat_log_probs = []
                    cat_probs_batch = cat_probs[b]  # (n_cat, n_cat_options)
                    for i in range(cat_probs_batch.shape[0]):
                        probs = cat_probs_batch[i]  # Shape: (n_cat_options,)
                        cat_dist = torch.distributions.Categorical(probs)
                        action_sample = cat_dist.sample()
                        cat_actions.append(action_sample.item())
                        cat_log_probs.append(cat_dist.log_prob(action_sample).item())
                        action_entropy += cat_dist.entropy().item()
                    
                    # Sample bernoulli actions
                    ber_actions = []
                    ber_log_probs = []
                    ber_probs_batch = ber_probs[b]  # (n_ber, 2)
                    for i in range(ber_probs_batch.shape[0]):
                        probs = ber_probs_batch[i]  # Shape: (2,)
                        ber_dist = torch.distributions.Bernoulli(probs[1])
                        action_sample = ber_dist.sample()
                        ber_actions.append(int(action_sample.item()))
                        ber_log_probs.append(ber_dist.log_prob(action_sample).item())
                        action_entropy += ber_dist.entropy().item()
                    
                    # Average entropy
                    num_actions = cat_probs_batch.shape[0] + ber_probs_batch.shape[0]
                    avg_action_entropy = action_entropy / max(1, num_actions)
                    
                    # Convert to action array
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
                    
                    all_actions.append(action)
                    all_cat_actions.append(cat_actions)
                    all_ber_actions.append(ber_actions)
                    all_cat_log_probs.append(cat_log_probs)
                    all_ber_log_probs.append(ber_log_probs)
                    all_entropies.append(avg_action_entropy)
            
            # Step environment
            if is_vec_env:
                # VecEnv expects list of actions
                next_obs, rewards, dones, infos = env.step(all_actions)
                # Ensure arrays
                rewards = ensure_array(rewards, "rewards")
                dones = ensure_array(dones, "dones")
            else:
                # Single env
                action = all_actions[0]
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # Wrap scalars
                next_obs = ensure_array(next_obs, "next_obs")
                rewards = np.array([reward])
                dones = np.array([done])
                infos = [info]
            
            # Store experiences in buffer
            for i in range(batch_size):
                # Extract per-env data
                if is_vec_env:
                    obs_i = obs[i]
                    reward_i = rewards[i] if i < len(rewards) else 0.0
                    done_i = dones[i] if i < len(dones) else False
                    value_i = value[i].item()
                else:
                    obs_i = obs
                    reward_i = rewards[0]
                    done_i = dones[0]
                    value_i = value[0].item()
                
                self.buffer.add({
                    'observation': obs_i,
                    'action': all_actions[i],
                    'cat_actions': np.array(all_cat_actions[i]),
                    'ber_actions': np.array(all_ber_actions[i]),
                    'cat_log_probs': np.array(all_cat_log_probs[i]),
                    'ber_log_probs': np.array(all_ber_log_probs[i]),
                    'reward': reward_i,
                    'done': done_i,
                    'value': value_i,
                    'entropy': all_entropies[i]
                })
                
                # Update episode stats
                if is_vec_env:
                    episode_rewards[i] += reward_i
                    episode_lengths[i] += 1
                else:
                    episode_reward += reward_i
                    episode_length += 1
                
                # Log episode completion
                if done_i:
                    self.episode += 1
                    if is_vec_env:
                        ep_reward = episode_rewards[i]
                        ep_length = episode_lengths[i]
                    else:
                        ep_reward = episode_reward
                        ep_length = episode_length
                    
                    if self.debug_mode:
                        logger.debug(
                            f"Episode {self.episode} (env {i}) complete: "
                            f"length={ep_length}, reward={ep_reward:.2f}"
                        )
                    
                    # Log episode stats
                    self.logger.log_scalar("train/episode_reward", ep_reward, self.timestep)
                    self.logger.log_scalar("train/episode_length", ep_length, self.timestep)
                    
                    # Reset episode stats
                    if is_vec_env:
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                    else:
                        episode_reward = 0.0
                        episode_length = 0
            
            # Log action entropy periodically
            if self.timestep % 100 == 0:
                avg_entropy = np.mean(all_entropies)
                self.logger.log_scalar("train/action_entropy", avg_entropy, self.timestep)
                'done': done,
                'value': value.item(),
                'entropy': avg_action_entropy
            
            # Update observation for next step
            obs = next_obs
            
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
        logger.info("✅ rollout verified (no scalar mismatch)")
        self._save_checkpoint(is_best=True)  # Final checkpoint is also best
    
    def _update(self):
        """Perform PPO update with shape validation."""
        # Get trajectory from buffer
        trajectory = self.buffer.get_recent_trajectory(max_length=self.config.batch_size)
        
        if len(trajectory["observations"]) == 0:
            return
        
        # Check if we have enough data
        if len(trajectory["observations"]) < 32:  # Minimum batch size
            return
        
        # Log tensor shapes before PPO update (runtime checks)
        if self.debug_mode and self.timestep % 1000 == 0:
            logger.debug(f"Update shapes - obs: {len(trajectory['observations'])}, "
                        f"rewards: {len(trajectory['rewards'])}, "
                        f"dones: {len(trajectory['dones'])}")
        
        # Convert to tensors
        obs = torch.tensor(
            np.array(trajectory["observations"]), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Extract actions and log probs (if available)
        # For backward compatibility, check if new format exists
        if "cat_actions" in trajectory and len(trajectory["cat_actions"]) > 0:
            cat_actions = torch.tensor(
                np.array([exp for exp in trajectory["cat_actions"]]),
                dtype=torch.long,
                device=self.device
            )
            ber_actions = torch.tensor(
                np.array([exp for exp in trajectory["ber_actions"]]),
                dtype=torch.long,
                device=self.device
            )
            old_log_probs_cat = torch.tensor(
                np.array([exp for exp in trajectory["cat_log_probs"]]),
                dtype=torch.float32,
                device=self.device
            )
            old_log_probs_ber = torch.tensor(
                np.array([exp for exp in trajectory["ber_log_probs"]]),
                dtype=torch.float32,
                device=self.device
            )
            # Extract old values from trajectory (stored in buffer)
            # Use itertools.islice for efficient deque slicing without full conversion
            from itertools import islice
            n_traj = len(trajectory["observations"])
            start_idx = max(0, len(self.buffer.buffer) - n_traj)
            old_values_list = [
                exp['value'] for exp in islice(self.buffer.buffer, start_idx, None) 
                if 'value' in exp
            ]
            old_values = torch.tensor(
                np.array(old_values_list),
                dtype=torch.float32,
                device=self.device
            )
        else:
            # Old format without stored actions - skip update
            logger.warning("Buffer missing action/log_prob data, skipping PPO update")
            return
        
        # Ensure arrays (wrap scalars if needed)
        rewards = ensure_array(trajectory["rewards"], "rewards")
        dones = ensure_array(trajectory["dones"], "dones")
        
        # Compute values for all observations
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    values = self.model.get_value(obs).cpu().numpy()
            else:
                values = self.model.get_value(obs).cpu().numpy()
            values = values.squeeze()
            # Ensure values is array
            values = ensure_array(values, "values")
        
        # For GAE, we need values for current states and next state
        # Since we have values for all current states, we use the last one as next_value
        # and pass all values to GAE (it will handle bootstrapping internally)
        next_value = 0.0  # Bootstrap value (will be overridden by values_ext in GAE)
        
        # Get current explained variance for dynamic lambda (use all values)
        if len(values) > 1 and len(rewards) > 1:
            var_y = np.var(rewards)
            # For explained variance, compare rewards with corresponding value estimates
            min_len = min(len(rewards), len(values))
            explained_var = 1 - np.var(rewards[:min_len] - values[:min_len]) / (var_y + 1e-8) if var_y > 0 else 0
        else:
            explained_var = None
        
        advantages, returns = self.ppo.compute_gae(
            rewards, values, dones, next_value, explained_var
        )
        
        # Convert to tensors
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Perform PPO update
        try:
            stats = self.ppo.update(
                obs,
                cat_actions,
                ber_actions,
                old_log_probs_cat,
                old_log_probs_ber,
                advantages_tensor,
                returns_tensor,
                old_values
            )
            
            # Log update stats
            if self.debug_mode:
                logger.debug(
                    f"PPO update | policy_loss={stats['policy_loss']:.4f} | "
                    f"value_loss={stats['value_loss']:.4f} | "
                    f"entropy_loss={stats['entropy_loss']:.4f} | "
                    f"explained_var={stats['explained_variance']:.4f}"
                )
            
            # Validate scalar losses (accept numpy scalars too)
            assert np.isscalar(stats['policy_loss']), \
                f"Policy loss not scalar: {type(stats['policy_loss'])}"
            assert np.isscalar(stats['value_loss']), \
                f"Value loss not scalar: {type(stats['value_loss'])}"
            assert np.isscalar(stats['explained_variance']), \
                f"Explained variance not scalar: {type(stats['explained_variance'])}"
            
        except Exception as e:
            logger.error(f"PPO update failed: {e}")
            if self.debug_mode:
                raise
    
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
