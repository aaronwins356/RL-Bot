"""
Utility functions for RL-Bot training.
Handles logging, device management, and checkpointing.
"""

import os
import torch
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(log_dir: str = "logs", verbose: bool = True) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        log_dir: Directory to save log files
        verbose: If True, also log to console
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("rl_bot")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate PyTorch device (CUDA/CPU).
    
    Args:
        device_str: Device specification ("auto", "cuda", "cpu")
        
    Returns:
        PyTorch device
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Using CPU (training will be slower)")
    
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt"
) -> str:
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        global_step: Current training step
        metrics: Dictionary of metrics (e.g., reward, elo)
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        
    Returns:
        Path to saved checkpoint
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / filename
    
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary containing metrics and step info
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class RewardNormalizer:
    """
    Normalizes rewards using running mean and standard deviation.
    """
    
    def __init__(self, clip_range: float = 10.0, epsilon: float = 1e-8):
        """
        Args:
            clip_range: Range to clip normalized rewards
            epsilon: Small value to avoid division by zero
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_range = clip_range
        self.epsilon = epsilon
    
    def normalize(self, reward: float) -> float:
        """
        Normalize a single reward value.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward
        """
        # Update statistics
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        
        # Normalize
        std = (self.var + self.epsilon) ** 0.5
        normalized = (reward - self.mean) / std
        
        # Clip
        normalized = max(-self.clip_range, min(self.clip_range, normalized))
        
        return normalized
    
    def reset(self):
        """Reset normalizer statistics."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


class ObservationNormalizer:
    """
    Normalizes observations using running mean and standard deviation.
    """
    
    def __init__(self, obs_dim: int, clip_range: float = 10.0, epsilon: float = 1e-8):
        """
        Args:
            obs_dim: Dimension of observation space
            clip_range: Range to clip normalized observations
            epsilon: Small value to avoid division by zero
        """
        self.mean = torch.zeros(obs_dim)
        self.var = torch.ones(obs_dim)
        self.count = 0
        self.clip_range = clip_range
        self.epsilon = epsilon
    
    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize observation tensor.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Normalized observation
        """
        # Update statistics
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        
        # Normalize
        std = torch.sqrt(self.var + self.epsilon)
        normalized = (obs - self.mean) / std
        
        # Clip
        normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
        
        return normalized
    
    def reset(self):
        """Reset normalizer statistics."""
        self.mean = torch.zeros_like(self.mean)
        self.var = torch.ones_like(self.var)
        self.count = 0


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
