"""Configuration management for RL-Bot.

This module handles loading and managing configuration from YAML files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration container."""
    
    # Training
    algorithm: str = "ppo"
    total_timesteps: int = 10000000
    batch_size: int = 4096
    n_epochs: int = 10
    learning_rate: float = 3e-4
    
    # Network
    network_architecture: str = "mlp"
    hidden_sizes: list = field(default_factory=lambda: [512, 512, 256])
    activation: str = "relu"
    use_lstm: bool = False
    
    # Policy
    policy_type: str = "hybrid"
    confidence_threshold: float = 0.7
    
    # Inference
    device: str = "cpu"
    frame_budget_ms: float = 8.0
    
    # Logging
    log_dir: str = "logs"
    tensorboard: bool = True
    log_interval: int = 1000
    
    # Checkpoints
    save_dir: str = "checkpoints"
    save_interval: int = 10000
    
    # Full config dictionary
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        # Extract top-level values
        training = config_dict.get("training", {})
        network = config_dict.get("network", {})
        policy = config_dict.get("policy", {})
        inference = config_dict.get("inference", {})
        logging_cfg = config_dict.get("logging", {})
        checkpoints = config_dict.get("checkpoints", {})
        
        return cls(
            algorithm=training.get("algorithm", "ppo"),
            total_timesteps=training.get("total_timesteps", 10000000),
            batch_size=training.get("batch_size", 4096),
            n_epochs=training.get("n_epochs", 10),
            learning_rate=training.get("learning_rate", 3e-4),
            network_architecture=network.get("architecture", "mlp"),
            hidden_sizes=network.get("hidden_sizes", [512, 512, 256]),
            activation=network.get("activation", "relu"),
            use_lstm=network.get("use_lstm", False),
            policy_type=policy.get("type", "hybrid"),
            confidence_threshold=policy.get("hybrid", {}).get("confidence_threshold", 0.7),
            device=inference.get("device", "cpu"),
            frame_budget_ms=inference.get("frame_budget_ms", 8.0),
            log_dir=logging_cfg.get("log_dir", "logs"),
            tensorboard=logging_cfg.get("tensorboard", True),
            log_interval=logging_cfg.get("log_interval", 1000),
            save_dir=checkpoints.get("save_dir", "checkpoints"),
            save_interval=logging_cfg.get("save_interval", 10000),
            raw_config=config_dict
        )


def load_config(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        overrides: Optional dictionary to override config values
        
    Returns:
        Config instance
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        config_dict = _deep_update(config_dict, overrides)
    
    return Config.from_dict(config_dict)


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep update dictionary (recursive merge).
    
    Args:
        base_dict: Base dictionary
        update_dict: Update dictionary
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Config, save_path: Path):
    """Save configuration to YAML file.
    
    Args:
        config: Config instance
        save_path: Path to save config
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config.raw_config, f, default_flow_style=False)
