"""Proximal Policy Optimization (PPO) algorithm implementation.

This module implements PPO for training the RL bot with:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function optimization
- Entropy regularization
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from core.models.nets import ActorCriticNet


class PPO:
    """Proximal Policy Optimization algorithm.
    
    Implements PPO with clipped objective for stable policy updates.
    """
    
    def __init__(
        self,
        model: ActorCriticNet,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize PPO.
        
        Args:
            model: Actor-critic network
            config: Training configuration
        """
        self.model = model
        self.config = config or {}
        
        # Hyperparameters
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.clip_range = self.config.get("clip_range", 0.2)
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.vf_coef = self.config.get("vf_coef", 0.5)
        self.ent_coef = self.config.get("ent_coef", 0.01)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.n_epochs = self.config.get("n_epochs", 10)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # Training stats
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "clip_fraction": [],
            "explained_variance": []
        }
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward array (T,)
            values: Value estimates (T,)
            dones: Done flags (T,)
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Append next value for bootstrap
        values_ext = np.append(values, next_value)
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value_t = 0
                last_gae = 0
            else:
                next_value_t = values_ext[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        observations: torch.Tensor,
        actions_cat: torch.Tensor,
        actions_ber: torch.Tensor,
        old_log_probs_cat: torch.Tensor,
        old_log_probs_ber: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """Perform PPO update.
        
        Args:
            observations: Batch of observations (B, obs_size)
            actions_cat: Categorical actions taken (B, n_cat)
            actions_ber: Bernoulli actions taken (B, n_ber)
            old_log_probs_cat: Old log probs for categorical actions (B, n_cat)
            old_log_probs_ber: Old log probs for bernoulli actions (B, n_ber)
            advantages: Computed advantages (B,)
            returns: Computed returns (B,)
            old_values: Old value estimates (B,)
            
        Returns:
            Dictionary with loss statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        epoch_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "clip_fraction": []
        }
        
        # Multiple epochs over the batch
        for epoch in range(self.n_epochs):
            # Forward pass
            cat_probs, ber_probs, values, _, _ = self.model(observations)
            
            # Compute new log probs
            new_log_probs_cat = self._compute_log_probs_categorical(
                cat_probs, actions_cat
            )
            new_log_probs_ber = self._compute_log_probs_bernoulli(
                ber_probs, actions_ber
            )
            
            # Compute ratio (pi_new / pi_old)
            ratio_cat = torch.exp(new_log_probs_cat - old_log_probs_cat)
            ratio_ber = torch.exp(new_log_probs_ber - old_log_probs_ber)
            ratio = ratio_cat.mean(dim=1) * ratio_ber.mean(dim=1)  # Combine ratios
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped)
            values = values.squeeze()
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_range,
                self.clip_range
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            # Entropy loss (for exploration)
            entropy_cat = self._compute_entropy_categorical(cat_probs)
            entropy_ber = self._compute_entropy_bernoulli(ber_probs)
            entropy_loss = -(entropy_cat + entropy_ber).mean()
            
            # Total loss
            total_loss = (
                policy_loss +
                self.vf_coef * value_loss +
                self.ent_coef * entropy_loss
            )
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track stats
            with torch.no_grad():
                clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
            
            epoch_stats["policy_loss"].append(policy_loss.item())
            epoch_stats["value_loss"].append(value_loss.item())
            epoch_stats["entropy_loss"].append(entropy_loss.item())
            epoch_stats["total_loss"].append(total_loss.item())
            epoch_stats["clip_fraction"].append(clip_fraction)
        
        # Average stats across epochs
        stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        
        # Compute explained variance
        with torch.no_grad():
            y_pred = values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            stats["explained_variance"] = (
                1 - np.var(y_true - y_pred) / (var_y + 1e-8)
                if var_y > 0 else 0
            )
        
        # Update training stats
        for k, v in stats.items():
            if k in self.training_stats:
                self.training_stats[k].append(v)
        
        return stats
    
    def _compute_log_probs_categorical(
        self,
        probs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for categorical actions.
        
        Args:
            probs: Action probabilities (B, n_cat, 3)
            actions: Actions taken (B, n_cat)
            
        Returns:
            Log probabilities (B, n_cat)
        """
        # Gather probabilities of actions taken
        batch_size = probs.shape[0]
        n_cat = probs.shape[1]
        
        # Create indices for gather
        indices = actions.unsqueeze(2)  # (B, n_cat, 1)
        
        # Gather action probabilities
        action_probs = torch.gather(probs, 2, indices).squeeze(2)  # (B, n_cat)
        
        # Compute log probs
        log_probs = torch.log(action_probs + 1e-8)
        
        return log_probs
    
    def _compute_log_probs_bernoulli(
        self,
        probs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for bernoulli actions.
        
        Args:
            probs: Action probabilities (B, n_ber, 2)
            actions: Actions taken (B, n_ber)
            
        Returns:
            Log probabilities (B, n_ber)
        """
        # Similar to categorical
        indices = actions.unsqueeze(2)
        action_probs = torch.gather(probs, 2, indices).squeeze(2)
        log_probs = torch.log(action_probs + 1e-8)
        
        return log_probs
    
    def _compute_entropy_categorical(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy for categorical distributions.
        
        Args:
            probs: Action probabilities (B, n_cat, 3)
            
        Returns:
            Entropy (B,)
        """
        # H = -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=2)  # (B, n_cat)
        return entropy.sum(dim=1)  # Sum over actions
    
    def _compute_entropy_bernoulli(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy for bernoulli distributions.
        
        Args:
            probs: Action probabilities (B, n_ber, 2)
            
        Returns:
            Entropy (B,)
        """
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=2)  # (B, n_ber)
        return entropy.sum(dim=1)
    
    def get_stats(self) -> Dict[str, List[float]]:
        """Get training statistics.
        
        Returns:
            Dictionary with training stats history
        """
        return self.training_stats
