"""
PPO training loop for Rocket League bot.
Simple and readable implementation with modular components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from rl_bot.core.model import ActorCritic
from rl_bot.core.utils import save_checkpoint, RewardNormalizer


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer.
    Implements the PPO algorithm with clipping and adaptive learning rate.
    """
    
    def __init__(
        self,
        model: ActorCritic,
        config: Dict[str, Any],
        device: torch.device,
        logger=None,
        tensorboard_writer: Optional[SummaryWriter] = None
    ):
        """
        Args:
            model: Actor-Critic model
            config: Training configuration
            device: PyTorch device
            logger: Logger instance
            tensorboard_writer: TensorBoard writer
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = tensorboard_writer
        
        # Training hyperparameters
        train_config = config.get('training', {})
        self.batch_size = train_config.get('batch_size', 4096)
        self.n_epochs = train_config.get('n_epochs', 10)
        self.learning_rate = train_config.get('learning_rate', 3e-4)
        self.gamma = train_config.get('gamma', 0.99)
        self.gae_lambda = train_config.get('gae_lambda', 0.95)
        self.clip_range = train_config.get('clip_range', 0.2)
        self.ent_coef = train_config.get('ent_coef', 0.01)
        self.vf_coef = train_config.get('vf_coef', 0.5)
        self.max_grad_norm = train_config.get('max_grad_norm', 0.5)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        if train_config.get('use_adaptive_lr', True):
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=train_config.get('total_timesteps', 10_000_000) // self.batch_size
            )
        else:
            self.scheduler = None
        
        # Normalization
        norm_config = config.get('normalization', {})
        self.normalize_rewards = norm_config.get('normalize_rewards', True)
        self.reward_normalizer = RewardNormalizer(
            clip_range=norm_config.get('clip_reward', 10.0)
        ) if self.normalize_rewards else None
        
        # Tracking
        self.global_step = 0
        self.update_count = 0
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards array [n_steps, n_envs]
            values: Value predictions [n_steps, n_envs]
            dones: Done flags [n_steps, n_envs]
            next_values: Next state values [n_envs]
            
        Returns:
            advantages: Advantage estimates [n_steps, n_envs]
            returns: Return estimates [n_steps, n_envs]
        """
        n_steps = len(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute advantages backwards
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = next_values
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            observations: Observations [batch_size, obs_dim]
            actions: Actions [batch_size]
            old_log_probs: Old log probabilities [batch_size]
            advantages: Advantage estimates [batch_size]
            returns: Return estimates [batch_size]
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        old_log_probs_tensor = torch.from_numpy(old_log_probs).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }
        
        # PPO epochs
        for epoch in range(self.n_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(observations))
            
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Mini-batch data
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.model.evaluate_actions(batch_obs, batch_actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['total_loss'] += loss.item()
                
                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    metrics['approx_kl'] += approx_kl
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    metrics['clip_fraction'] += clip_fraction
        
        # Average metrics
        num_updates = self.n_epochs * (len(observations) // self.batch_size)
        for key in metrics:
            metrics[key] /= num_updates
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        self.update_count += 1
        
        return metrics
    
    def train_step(
        self,
        env,
        n_steps: int,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Collect experience and perform one training step.
        
        Args:
            env: Vectorized environment
            n_steps: Number of steps to collect
            deterministic: Use deterministic policy
            
        Returns:
            Training metrics and episode statistics
        """
        # Storage
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Reset if needed
        if not hasattr(self, 'current_obs'):
            self.current_obs, _ = env.reset()
        
        # Collect experience
        for step in range(n_steps):
            obs_tensor = torch.from_numpy(self.current_obs).float().to(self.device)
            
            # Get action
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor, deterministic)
            
            # Store
            observations.append(self.current_obs)
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated | truncated
            
            # Normalize rewards
            if self.normalize_rewards:
                normalized_reward = np.array([self.reward_normalizer.normalize(r) for r in reward])
                rewards.append(normalized_reward)
            else:
                rewards.append(reward)
            
            dones.append(done)
            
            # Update observation
            self.current_obs = next_obs
            self.global_step += env.num_envs
            
            # Reset environments that are done
            # (handled automatically by vectorized env)
        
        # Convert to arrays
        observations = np.array(observations)
        actions = np.array(actions)
        log_probs = np.array(log_probs)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        
        # Get final values
        with torch.no_grad():
            next_obs_tensor = torch.from_numpy(self.current_obs).float().to(self.device)
            next_values = self.model.get_value(next_obs_tensor).cpu().numpy()
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        # Flatten arrays for update
        observations_flat = observations.reshape(-1, observations.shape[-1])
        actions_flat = actions.reshape(-1)
        log_probs_flat = log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
        # Perform update
        train_metrics = self.update(
            observations_flat,
            actions_flat,
            log_probs_flat,
            advantages_flat,
            returns_flat
        )
        
        # Add episode statistics
        train_metrics['mean_reward'] = rewards.mean()
        train_metrics['mean_value'] = values.mean()
        train_metrics['global_step'] = self.global_step
        
        return train_metrics


def train(
    env,
    config: Dict[str, Any],
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    logger=None,
    resume_from: Optional[str] = None
) -> ActorCritic:
    """
    Main training function.
    
    Args:
        env: Training environment
        config: Configuration dictionary
        total_timesteps: Total training timesteps
        device: PyTorch device
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        logger: Logger instance
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained model
    """
    # Create model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    network_config = config.get('network', {})
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=network_config.get('hidden_sizes', [512, 512, 256]),
        activation=network_config.get('activation', 'relu'),
        use_layer_norm=network_config.get('use_layer_norm', False)
    ).to(device)
    
    if logger:
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Create trainer
    trainer = PPOTrainer(model, config, device, logger, writer)
    
    # Resume from checkpoint if specified
    if resume_from:
        from rl_bot.core.utils import load_checkpoint
        checkpoint = load_checkpoint(resume_from, model, trainer.optimizer, device)
        trainer.global_step = checkpoint.get('global_step', 0)
        if logger:
            logger.info(f"Resumed from checkpoint: {resume_from} (step {trainer.global_step})")
    
    # Training loop
    train_config = config.get('training', {})
    n_steps = train_config.get('n_steps', 2048)
    log_interval = config.get('logging', {}).get('log_interval', 10)
    save_interval = config.get('logging', {}).get('save_interval', 500)
    
    best_reward = -float('inf')
    
    pbar = tqdm(total=total_timesteps, desc="Training")
    pbar.update(trainer.global_step)
    
    while trainer.global_step < total_timesteps:
        # Training step
        metrics = trainer.train_step(env, n_steps)
        
        # Update progress bar
        pbar.update(n_steps * env.num_envs)
        pbar.set_postfix({
            'reward': f"{metrics['mean_reward']:.2f}",
            'value': f"{metrics['mean_value']:.2f}",
            'loss': f"{metrics['total_loss']:.4f}"
        })
        
        # Log metrics
        if trainer.update_count % log_interval == 0:
            for key, value in metrics.items():
                writer.add_scalar(f'train/{key}', value, trainer.global_step)
            
            if logger:
                logger.info(
                    f"Step {trainer.global_step}: "
                    f"Reward={metrics['mean_reward']:.2f}, "
                    f"Loss={metrics['total_loss']:.4f}, "
                    f"KL={metrics['approx_kl']:.4f}"
                )
        
        # Save checkpoint
        if trainer.update_count % save_interval == 0:
            checkpoint_path = save_checkpoint(
                model,
                trainer.optimizer,
                trainer.global_step,
                {'mean_reward': metrics['mean_reward']},
                checkpoint_dir,
                filename=f"checkpoint_{trainer.global_step}.pt"
            )
            
            # Save best model
            if metrics['mean_reward'] > best_reward:
                best_reward = metrics['mean_reward']
                save_checkpoint(
                    model,
                    trainer.optimizer,
                    trainer.global_step,
                    {'mean_reward': best_reward},
                    checkpoint_dir,
                    filename="best_model.pt"
                )
                if logger:
                    logger.info(f"New best model saved with reward: {best_reward:.2f}")
    
    pbar.close()
    writer.close()
    
    # Save final model
    save_checkpoint(
        model,
        trainer.optimizer,
        trainer.global_step,
        {'mean_reward': metrics['mean_reward']},
        checkpoint_dir,
        filename="final_model.pt"
    )
    
    if logger:
        logger.info("Training complete!")
    
    return model
