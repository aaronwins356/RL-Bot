"""
Enhanced PyTorch neural networks for PPO with modern optimizations.
Includes torch.compile() support and optional LSTM for temporal awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with modern features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = False,
        orthogonal_init: bool = True
    ):
        super().__init__()
        
        # Activation function
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
            in_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Orthogonal initialization
        if orthogonal_init:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RecurrentPolicy(nn.Module):
    """Recurrent policy network with LSTM for temporal awareness."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True
        )
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]
            hidden_state: Optional (h, c) tuple
            
        Returns:
            output: LSTM output
            hidden_state: New (h, c) tuple
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        output, hidden_state = self.lstm(x, hidden_state)
        
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        return output.squeeze(1), hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Supports both discrete and continuous action spaces.
    Optional LSTM for temporal awareness.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: dict
    ):
        super().__init__()
        
        network_config = config.get('network', {})
        hidden_sizes = network_config.get('hidden_sizes', [512, 512, 256])
        activation = network_config.get('activation', 'relu')
        use_layer_norm = network_config.get('use_layer_norm', False)
        orthogonal_init = network_config.get('orthogonal_init', True)
        use_lstm = network_config.get('use_lstm', False)
        lstm_hidden_size = network_config.get('lstm_hidden_size', 256)
        
        self.use_lstm = use_lstm
        
        # Feature extractor (shared or separate)
        if use_lstm:
            self.lstm = RecurrentPolicy(obs_dim, lstm_hidden_size, use_layer_norm=use_layer_norm)
            feature_dim = lstm_hidden_size
        else:
            self.feature_extractor = MLP(
                obs_dim,
                hidden_sizes[:-1],
                hidden_sizes[-1],
                activation,
                use_layer_norm,
                orthogonal_init
            )
            feature_dim = hidden_sizes[-1]
        
        # Policy head (actor)
        self.policy_head = nn.Linear(feature_dim, action_dim)
        
        # Value head (critic)
        self.value_head = nn.Linear(feature_dim, 1)
        
        # Initialize heads
        if orthogonal_init:
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
            nn.init.constant_(self.policy_head.bias, 0.0)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            obs: Observations [batch, obs_dim]
            hidden_state: Optional LSTM hidden state
            
        Returns:
            logits: Action logits [batch, action_dim]
            value: State value [batch, 1]
            hidden_state: New LSTM hidden state (if using LSTM)
        """
        if self.use_lstm:
            features, hidden_state = self.lstm(obs, hidden_state)
        else:
            features = self.feature_extractor(obs)
            hidden_state = None
        
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return logits, value, hidden_state
    
    def get_value(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Get state value only."""
        _, value, _ = self.forward(obs, hidden_state)
        return value.squeeze(-1)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value.
        
        Args:
            obs: Observations
            action: Optional action to evaluate
            hidden_state: Optional LSTM hidden state
            deterministic: If True, use greedy action selection
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
        """
        logits, value, _ = self.forward(obs, hidden_state)
        
        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


def create_actor_critic(obs_dim: int, action_dim: int, config: dict) -> ActorCritic:
    """Factory function to create ActorCritic model."""
    model = ActorCritic(obs_dim, action_dim, config)
    
    # Apply torch.compile if enabled (PyTorch 2.x)
    if config.get('network', {}).get('use_torch_compile', False):
        try:
            model = torch.compile(model)
            print("✓ torch.compile() enabled for performance boost")
        except Exception as e:
            print(f"⚠ torch.compile() failed: {e}")
            print("  Continuing without compilation")
    
    return model
