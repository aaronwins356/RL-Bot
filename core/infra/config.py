"""Configuration management for RL-Bot.

This module handles loading and managing configuration from YAML files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


class Config:
    """Configuration container with both dict-style and dot-notation access.
    
    This class wraps OmegaConf to provide flexible configuration access.
    Supports both config['training']['algorithm'] and config.training.algorithm
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        # Store raw config
        self._raw_config = config_dict
        
        # Create OmegaConf for dot-notation access
        self._omega_config = OmegaConf.create(config_dict)
        
        # Also support legacy flat attributes for backward compatibility
        self._init_legacy_attributes()
    
    def _init_legacy_attributes(self):
        """Initialize legacy flat attributes for backward compatibility."""
        training = self._raw_config.get("training", {})
        network = self._raw_config.get("network", {})
        policy = self._raw_config.get("policy", {})
        inference = self._raw_config.get("inference", {})
        logging_cfg = self._raw_config.get("logging", {})
        checkpoints = self._raw_config.get("checkpoints", {})
        
        # Training attributes
        self.algorithm = training.get("algorithm", "ppo")
        self.total_timesteps = training.get("total_timesteps", 10000000)
        self.batch_size = training.get("batch_size", 4096)
        self.n_epochs = training.get("n_epochs", 10)
        self.learning_rate = training.get("learning_rate", 3e-4)
        
        # Network attributes
        self.network_architecture = network.get("architecture", "mlp")
        self.hidden_sizes = network.get("hidden_sizes", [512, 512, 256])
        self.activation = network.get("activation", "relu")
        self.use_lstm = network.get("use_lstm", False)
        
        # Policy attributes
        self.policy_type = policy.get("type", "hybrid")
        self.confidence_threshold = policy.get("hybrid", {}).get("confidence_threshold", 0.7)
        
        # Inference attributes
        self.device = inference.get("device", "cpu")
        self.frame_budget_ms = inference.get("frame_budget_ms", 8.0)
        
        # Logging attributes
        self.log_dir = logging_cfg.get("log_dir", "logs")
        self.tensorboard = logging_cfg.get("tensorboard", True)
        self.log_interval = logging_cfg.get("log_interval", 1000)
        
        # Checkpoint attributes
        self.save_dir = checkpoints.get("save_dir", "checkpoints")
        self.save_interval = logging_cfg.get("save_interval", 10000)
        
        # Raw config for backward compatibility
        self.raw_config = self._raw_config
    
    def __getitem__(self, key: str) -> Any:
        """Get config value by key (dict-style access).
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self._raw_config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set config value by key (dict-style access).
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._raw_config[key] = value
        # Update OmegaConf as well
        OmegaConf.update(self._omega_config, key, value, merge=False)
    
    def __getattr__(self, name: str) -> Any:
        """Get config value by attribute (dot-notation access).
        
        Args:
            name: Attribute name
            
        Returns:
            Configuration value
        """
        # Avoid infinite recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Try OmegaConf first for nested access
        try:
            return getattr(self._omega_config, name)
        except AttributeError:
            # Fall back to raw config
            if name in self._raw_config:
                return self._raw_config[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._raw_config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._raw_config.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(config_dict)


class ConfigManager:
    """Manager for loading, validating, and managing configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            config_path: Path to config file (optional)
        """
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.config = self.load(config_path)
    
    def load(self, config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
        """Load configuration from file.
        
        Args:
            config_path: Path to config file
            overrides: Optional overrides
            
        Returns:
            Config instance
        """
        self.config_path = config_path
        self.config = load_config(config_path, overrides)
        return self.config
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> Config:
        """Apply overrides to loaded config.
        
        Args:
            overrides: Override dictionary
            
        Returns:
            Updated config
        """
        if self.config is None:
            raise ValueError("No config loaded. Call load() first.")
        
        config_dict = self.config.to_dict()
        config_dict = _deep_update(config_dict, overrides)
        self.config = Config(config_dict)
        return self.config
    
    def validate_schema(self) -> bool:
        """Validate config against expected schema.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if self.config is None:
            raise ValueError("No config loaded. Call load() first.")
        
        # Check required top-level keys
        required_keys = ["training", "network", "policy", "inference", "logging"]
        config_dict = self.config.to_dict()
        
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        # Check required training keys
        training = config_dict.get("training", {})
        required_training = ["algorithm", "total_timesteps"]
        missing_training = [key for key in required_training if key not in training]
        if missing_training:
            raise ValueError(f"Missing required training config keys: {missing_training}")
        
        return True
    
    def get_safe(self, key_path: str, default: Any = None) -> Any:
        """Safely get nested config value with dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., "training.algorithm")
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        if self.config is None:
            return default
        
        try:
            keys = key_path.split('.')
            value = self.config.to_dict()
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


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
