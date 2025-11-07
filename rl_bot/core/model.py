"""
PyTorch neural network models for PPO.
Includes policy network (actor) and value network (critic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_sizes: List of hidden layer sizes
            output_dim: Output dimension
            activation: Activation function name
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        in_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Shares some layers between policy and value networks.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [512, 512, 256],
        activation: str = "relu",
        use_layer_norm: bool = False,
        shared_layers: int = 0
    ):
        """
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension (number of discrete actions)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            shared_layers: Number of initial shared layers (0 = separate networks)
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extractor (if any)
        if shared_layers > 0:
            shared_hidden = hidden_sizes[:shared_layers]
            self.shared_net = MLP(
                obs_dim, shared_hidden, shared_hidden[-1],
                activation=activation, use_layer_norm=use_layer_norm
            )
            policy_input_dim = shared_hidden[-1]
            value_input_dim = shared_hidden[-1]
            remaining_hidden = hidden_sizes[shared_layers:]
        else:
            self.shared_net = None
            policy_input_dim = obs_dim
            value_input_dim = obs_dim
            remaining_hidden = hidden_sizes
        
        # Policy network (actor) - outputs action logits
        self.policy_net = MLP(
            policy_input_dim,
            remaining_hidden,
            action_dim,
            activation=activation,
            use_layer_norm=use_layer_norm
        )
        
        # Value network (critic) - outputs state value
        self.value_net = MLP(
            value_input_dim,
            remaining_hidden,
            1,
            activation=activation,
            use_layer_norm=use_layer_norm
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            
        Returns:
            action_logits: Logits for action distribution [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        # Shared features
        if self.shared_net is not None:
            features = self.shared_net(obs)
        else:
            features = obs
        
        # Policy and value
        action_logits = self.policy_net(features)
        value = self.value_net(features)
        
        return action_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            deterministic: If True, return argmax action
            
        Returns:
            action: Sampled actions [batch_size]
            log_prob: Log probability of actions [batch_size]
            value: State value estimate [batch_size]
        """
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        # Sample or take argmax
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.
        Used during PPO updates.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Action tensor [batch_size]
            
        Returns:
            log_prob: Log probability of actions [batch_size]
            value: State value estimate [batch_size]
            entropy: Entropy of action distribution [batch_size]
        """
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        # Evaluate
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, value.squeeze(-1), entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for observations.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            
        Returns:
            value: State value estimate [batch_size]
        """
        if self.shared_net is not None:
            features = self.shared_net(obs)
        else:
            features = obs
        
        value = self.value_net(features)
        return value.squeeze(-1)


class PPOAgent:
    """
    PPO agent that wraps the ActorCritic network and handles action selection.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [512, 512, 256],
        activation: str = "relu",
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            device: PyTorch device
        """
        self.device = device
        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation
        ).to(device)
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict actions for given observations.
        
        Args:
            obs: Observation array [batch_size, obs_dim] or [obs_dim]
            deterministic: If True, return argmax action
            
        Returns:
            actions: Action array [batch_size] or scalar
            values: Value array [batch_size] or scalar
        """
        # Convert to tensor
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        
        # Get action
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(obs_tensor, deterministic)
        
        # Convert to numpy
        action_np = action.cpu().numpy()
        value_np = value.cpu().numpy()
        
        if squeeze:
            action_np = action_np[0]
            value_np = value_np[0]
        
        return action_np, value_np
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
