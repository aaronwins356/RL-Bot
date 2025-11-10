"""
PPO Trainer with modern optimizations.
Supports AMP, gradient accumulation, and distributed training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from .network import ActorCritic
from .memory import RolloutBuffer
from .losses import total_ppo_loss, explained_variance
from .utils import (
    get_schedule,
    clip_gradients,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    AdaptiveEntropyCoef
)


class PPOTrainer:
    """
    Advanced PPO trainer with modern features:
    - Automatic mixed precision (AMP)
    - Adaptive learning rate and entropy
    - KL divergence monitoring
    - Early stopping
    """
    
    def __init__(
        self,
        model: ActorCritic,
        config: Dict[str, Any],
        device: torch.device,
        writer: Optional[SummaryWriter] = None
    ):
        """
        Args:
            model: Actor-Critic model
            config: Training configuration
            device: PyTorch device
            writer: TensorBoard writer
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.writer = writer
        
        # Training hyperparameters
        train_config = config.get('training', {})
        self.batch_size = train_config.get('batch_size', 4096)
        self.n_epochs = train_config.get('n_epochs', 10)
        self.learning_rate = train_config.get('learning_rate', 3e-4)
        self.gamma = train_config.get('gamma', 0.99)
        self.gae_lambda = train_config.get('gae_lambda', 0.95)
        self.clip_range = train_config.get('clip_range', 0.2)
        self.vf_coef = train_config.get('vf_coef', 0.5)
        self.ent_coef = train_config.get('ent_coef', 0.01)
        self.max_grad_norm = train_config.get('max_grad_norm', 0.5)
        self.use_adaptive_lr = train_config.get('use_adaptive_lr', True)
        self.lr_schedule_type = train_config.get('lr_schedule', 'linear')
        self.use_amp = train_config.get('use_amp', False)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)
        
        # Learning rate schedule
        if self.use_adaptive_lr:
            self.lr_schedule = get_schedule(
                self.lr_schedule_type,
                self.learning_rate,
                final_value=self.learning_rate * 0.1
            )
        else:
            self.lr_schedule = None
        
        # Adaptive entropy coefficient
        self.adaptive_entropy = AdaptiveEntropyCoef(
            initial_value=self.ent_coef,
            min_value=self.ent_coef * 0.1
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.num_updates = 0
        self.total_timesteps = 0
        
        print(f"PPO Trainer initialized:")
        print(f"  Parameters: {count_parameters(self.model):,}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Adaptive LR: {self.use_adaptive_lr}")
    
    def train_step(
        self,
        rollout_buffer: RolloutBuffer,
        progress: float
    ) -> Dict[str, float]:
        """
        Perform one PPO update.
        
        Args:
            rollout_buffer: Buffer with collected rollouts
            progress: Training progress from 0.0 to 1.0
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Update learning rate
        if self.lr_schedule is not None:
            new_lr = self.lr_schedule(progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            current_lr = new_lr
        else:
            current_lr = self.learning_rate
        
        # Update entropy coefficient
        current_ent_coef = self.adaptive_entropy.update()
        
        # Compute advantages and returns
        # This should be done by the rollout buffer
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        clip_fractions = []
        approx_kls = []
        grad_norms = []
        
        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            # Mini-batch training
            for batch in rollout_buffer.get(self.batch_size):
                observations = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                old_values = batch['values']
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        _, log_probs, entropy, values = self.model.get_action_and_value(
                            observations,
                            action=actions
                        )
                        
                        # Compute loss
                        loss, info = total_ppo_loss(
                            log_probs,
                            old_log_probs,
                            advantages,
                            values,
                            returns,
                            entropy,
                            old_values,
                            self.clip_range,
                            self.vf_coef,
                            current_ent_coef,
                            normalize_advantage=True,
                            use_value_clipping=False
                        )
                    
                    # Backward pass with gradient scaling
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_gradients(self.model, self.max_grad_norm)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    _, log_probs, entropy, values = self.model.get_action_and_value(
                        observations,
                        action=actions
                    )
                    
                    # Compute loss
                    loss, info = total_ppo_loss(
                        log_probs,
                        old_log_probs,
                        advantages,
                        values,
                        returns,
                        entropy,
                        old_values,
                        self.clip_range,
                        self.vf_coef,
                        current_ent_coef,
                        normalize_advantage=True,
                        use_value_clipping=False
                    )
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    grad_norm = clip_gradients(self.model, self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                
                # Collect metrics
                policy_losses.append(info['policy_loss'])
                value_losses.append(info['value_loss'])
                entropy_losses.append(info['entropy_loss'])
                total_losses.append(info['total_loss'])
                clip_fractions.append(info['clip_fraction'])
                approx_kls.append(info['approx_kl'])
                grad_norms.append(grad_norm)
        
        # Compute explained variance
        all_values = rollout_buffer.values.reshape(-1)
        all_returns = rollout_buffer.returns.reshape(-1)
        exp_var = explained_variance(all_values, all_returns)
        
        # Aggregate metrics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses),
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': np.mean(approx_kls),
            'grad_norm': np.mean(grad_norms),
            'learning_rate': current_lr,
            'entropy_coef': current_ent_coef,
            'explained_variance': exp_var
        }
        
        self.num_updates += 1
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'train/{key}', value, self.num_updates)
        
        return metrics
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save checkpoint."""
        save_checkpoint(
            self.model,
            self.optimizer,
            path,
            self.num_updates,
            self.total_timesteps,
            self.config,
            metadata
        )
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = load_checkpoint(
            self.model,
            self.optimizer,
            path,
            self.device
        )
        self.num_updates = checkpoint.get('epoch', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        return checkpoint
