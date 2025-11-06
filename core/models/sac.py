"""Soft Actor-Critic (SAC) algorithm implementation.

This module implements SAC for continuous action space training with:
- Maximum entropy RL framework
- Twin Q-networks for value estimation
- Automatic entropy temperature tuning
- Replay buffer support
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SACCritic(nn.Module):
    """Twin Q-networks for SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list = [256, 256]):
        """Initialize SAC critic.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
        """
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Tuple of (Q1 value, Q2 value)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACActor(nn.Module):
    """Stochastic policy network for SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list = [256, 256]):
        """Initialize SAC actor.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], action_dim)
        
        # Action bounds
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get mean and log_std.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log prob with change of variables
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # Scale action
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob


class SAC:
    """Soft Actor-Critic algorithm.
    
    Implements SAC with automatic entropy temperature tuning.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ):
        """Initialize SAC.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension  
            config: Training configuration
            device: Device for training
        """
        self.config = config or {}
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.tau = self.config.get("tau", 0.005)  # Target network soft update
        self.alpha = self.config.get("alpha", 0.2)  # Entropy temperature
        self.auto_entropy_tuning = self.config.get("auto_entropy_tuning", True)
        
        # Networks
        hidden_sizes = self.config.get("hidden_sizes", [256, 256])
        
        self.actor = SACActor(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic = SACCritic(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = SACCritic(obs_dim, action_dim, hidden_sizes).to(device)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        
        # Training stats
        self.training_stats = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha": [],
            "entropy": []
        }
        self.update_count = 0
        
        logger.info(f"SAC initialized with obs_dim={obs_dim}, action_dim={action_dim}")
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from policy.
        
        Args:
            obs: Observation
            deterministic: Use deterministic policy
            
        Returns:
            Action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if deterministic:
                mean, _ = self.actor(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(obs_tensor)
            
            return action.cpu().numpy()[0]
    
    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor
    ) -> Dict[str, float]:
        """Update SAC networks.
        
        Args:
            obs: Observation batch
            action: Action batch
            reward: Reward batch
            next_obs: Next observation batch
            done: Done flags batch
            
        Returns:
            Dictionary of losses
        """
        self.update_count += 1
        
        # Update critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            
            value_target = reward + (1 - done) * self.gamma * (q_next - self.alpha * next_log_prob)
        
        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_action, log_prob = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        self._soft_update_target()
        
        # Track stats
        self.training_stats["actor_loss"].append(actor_loss.item())
        self.training_stats["critic_loss"].append(critic_loss.item())
        self.training_stats["alpha"].append(self.alpha)
        self.training_stats["entropy"].append(-log_prob.mean().item())
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha,
            "entropy": -log_prob.mean().item()
        }
    
    def _soft_update_target(self):
        """Soft update target network parameters."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.training_stats["actor_loss"]:
            return {}
        
        return {
            "actor_loss": np.mean(self.training_stats["actor_loss"][-100:]),
            "critic_loss": np.mean(self.training_stats["critic_loss"][-100:]),
            "alpha": self.alpha,
            "entropy": np.mean(self.training_stats["entropy"][-100:])
        }
    
    def save(self, path: str):
        """Save model state.
        
        Args:
            path: Save path
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None
        }, path)
    
    def load(self, path: str):
        """Load model state.
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.auto_entropy_tuning and checkpoint.get('log_alpha') is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
