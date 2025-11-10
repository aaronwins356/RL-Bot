"""
Utility functions for PPO core.
Includes learning rate scheduling, gradient clipping, and checkpointing.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_str: "auto", "cuda", or "cpu"
        
    Returns:
        device: PyTorch device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LinearSchedule:
    """Linear learning rate schedule."""
    
    def __init__(self, initial_value: float, final_value: float = 0.0):
        self.initial_value = initial_value
        self.final_value = final_value
    
    def __call__(self, progress: float) -> float:
        """
        Args:
            progress: Training progress from 0.0 to 1.0
            
        Returns:
            value: Scheduled value
        """
        return self.initial_value + progress * (self.final_value - self.initial_value)


class ExponentialSchedule:
    """Exponential learning rate schedule."""
    
    def __init__(self, initial_value: float, decay_rate: float = 0.99):
        self.initial_value = initial_value
        self.decay_rate = decay_rate
    
    def __call__(self, progress: float) -> float:
        return self.initial_value * (self.decay_rate ** progress)


class CosineAnnealingSchedule:
    """Cosine annealing learning rate schedule."""
    
    def __init__(self, initial_value: float, final_value: float = 0.0):
        self.initial_value = initial_value
        self.final_value = final_value
    
    def __call__(self, progress: float) -> float:
        return self.final_value + (self.initial_value - self.final_value) * \
               0.5 * (1.0 + np.cos(np.pi * progress))


def get_schedule(schedule_type: str, initial_value: float, final_value: float = 0.0):
    """
    Get learning rate schedule.
    
    Args:
        schedule_type: "linear", "exponential", "cosine", or "constant"
        initial_value: Initial value
        final_value: Final value
        
    Returns:
        schedule: Schedule function
    """
    if schedule_type == "linear":
        return LinearSchedule(initial_value, final_value)
    elif schedule_type == "exponential":
        return ExponentialSchedule(initial_value)
    elif schedule_type == "cosine":
        return CosineAnnealingSchedule(initial_value, final_value)
    elif schedule_type == "constant":
        return lambda progress: initial_value
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def clip_gradients(model: nn.Module, max_grad_norm: float) -> float:
    """
    Clip gradients by norm.
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum gradient norm
        
    Returns:
        grad_norm: Gradient norm before clipping
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm
    )
    return grad_norm.item()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    epoch: int,
    total_timesteps: int,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        path: Save path
        epoch: Current epoch
        total_timesteps: Total timesteps trained
        config: Training configuration
        metadata: Additional metadata
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'total_timesteps': total_timesteps,
        'config': config
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        path: Checkpoint path
        device: Device to load to
        
    Returns:
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningMeanStd:
    """
    Running mean and standard deviation tracker.
    Useful for normalizing observations or rewards.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class RewardNormalizer:
    """Normalize rewards with running statistics."""
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_mean_std = RunningMeanStd(shape=())
        self.returns = 0.0
    
    def normalize(self, reward: float, done: bool) -> float:
        """Normalize reward and update statistics."""
        self.returns = reward + self.gamma * self.returns * (1.0 - done)
        self.running_mean_std.update(np.array([self.returns]))
        
        normalized = reward / np.sqrt(self.running_mean_std.var + self.epsilon)
        
        if done:
            self.returns = 0.0
        
        return normalized
    
    def reset(self):
        """Reset returns."""
        self.returns = 0.0


class AdaptiveEntropyCoef:
    """
    Adaptive entropy coefficient that decays based on performance.
    Higher entropy early on for exploration, lower later for exploitation.
    """
    
    def __init__(
        self,
        initial_value: float = 0.01,
        min_value: float = 0.001,
        decay_rate: float = 0.99
    ):
        self.initial_value = initial_value
        self.min_value = min_value
        self.decay_rate = decay_rate
        self.current_value = initial_value
    
    def update(self, performance_metric: Optional[float] = None) -> float:
        """
        Update entropy coefficient.
        
        Args:
            performance_metric: Optional metric to guide adaptation
            
        Returns:
            current_value: Current entropy coefficient
        """
        # Simple decay
        self.current_value = max(
            self.min_value,
            self.current_value * self.decay_rate
        )
        
        return self.current_value
    
    def get(self) -> float:
        """Get current value."""
        return self.current_value


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score (higher is better)
            
        Returns:
            should_stop: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
