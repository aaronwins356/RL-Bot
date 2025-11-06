"""Neural network architectures for RL policies.

This module defines various network architectures:
- MLPNet: Multi-layer perceptron
- CNNLSTMNet: CNN + LSTM for temporal features
- ActorCriticNet: Combined actor-critic architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class MLPNet(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu"
    ):
        """Initialize MLP network.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output
            activation: Activation function ("relu", "tanh", "elu")
        """
        super().__init__()
        
        self.activation_name = activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class CNNLSTMNet(nn.Module):
    """CNN + LSTM network for temporal features."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lstm_hidden_size: int,
        output_size: int,
        num_lstm_layers: int = 1
    ):
        """Initialize CNN-LSTM network.
        
        Args:
            input_size: Size of input features
            hidden_size: Hidden size for MLP
            lstm_hidden_size: Hidden size for LSTM
            output_size: Size of output
            num_lstm_layers: Number of LSTM layers
        """
        super().__init__()
        
        # Feature extraction
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            hidden_size,
            lstm_hidden_size,
            num_lstm_layers,
            batch_first=True
        )
        
        # Output
        self.fc_out = nn.Linear(lstm_hidden_size, output_size)
        
        self.hidden_state = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional LSTM hidden state
            
        Returns:
            Tuple of (output, hidden_state)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1] if len(x.shape) > 2 else 1
        
        # Reshape if necessary
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # LSTM
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        
        # Output (take last timestep)
        x = x[:, -1, :]
        x = self.fc_out(x)
        
        return x, hidden
    
    def reset_hidden(self):
        """Reset LSTM hidden state."""
        self.hidden_state = None


class ActorCriticNet(nn.Module):
    """Actor-Critic network for PPO/SAC.
    
    Outputs both action distribution parameters and value estimate.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        action_categoricals: int,
        action_bernoullis: int,
        activation: str = "relu",
        use_lstm: bool = False,
        lstm_hidden_size: int = 256
    ):
        """Initialize Actor-Critic network.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            action_categoricals: Number of categorical actions (e.g., 5 for throttle, steer, etc.)
            action_bernoullis: Number of binary actions (e.g., 3 for jump, boost, handbrake)
            activation: Activation function
            use_lstm: Whether to use LSTM for temporal processing
            lstm_hidden_size: Hidden size for LSTM
        """
        super().__init__()
        
        self.action_categoricals = action_categoricals
        self.action_bernoullis = action_bernoullis
        self.use_lstm = use_lstm
        
        if use_lstm:
            # CNN-LSTM backbone
            self.backbone = CNNLSTMNet(
                input_size,
                hidden_sizes[0],
                lstm_hidden_size,
                hidden_sizes[-1]
            )
            feature_size = hidden_sizes[-1]
        else:
            # MLP backbone
            self.backbone = MLPNet(
                input_size,
                hidden_sizes,
                hidden_sizes[-1],
                activation
            )
            feature_size = hidden_sizes[-1]
        
        # Actor heads
        # Categorical actions: each has 3 options (-1, 0, 1)
        self.cat_heads = nn.Linear(feature_size, 3 * action_categoricals)
        
        # Bernoulli actions: each has 2 options (0, 1)
        self.ber_heads = nn.Linear(feature_size, 2 * action_bernoullis)
        
        # Critic head (value function)
        self.value_head = nn.Linear(feature_size, 1)
        
        # Confidence head (entropy-based)
        self.confidence_head = nn.Linear(feature_size, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass.
        
        Args:
            x: Input tensor
            hidden: Optional LSTM hidden state
            
        Returns:
            Tuple of (cat_probs, ber_probs, value, confidence, hidden)
        """
        # Backbone
        if self.use_lstm:
            features, hidden = self.backbone(x, hidden)
        else:
            features = self.backbone(x)
            hidden = None
        
        # Actor outputs
        cat_logits = self.cat_heads(features)
        cat_probs = F.softmax(
            cat_logits.view(-1, self.action_categoricals, 3),
            dim=2
        )
        
        ber_logits = self.ber_heads(features)
        ber_probs = F.softmax(
            ber_logits.view(-1, self.action_bernoullis, 2),
            dim=2
        )
        
        # Critic output
        value = self.value_head(features)
        
        # Confidence output (sigmoid to get 0-1 range)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return cat_probs, ber_probs, value, confidence, hidden
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate only (for critic updates).
        
        Args:
            x: Input tensor
            
        Returns:
            Value estimate
        """
        if self.use_lstm:
            features, _ = self.backbone(x)
        else:
            features = self.backbone(x)
        
        return self.value_head(features)
