"""Reward mode configurations for flexible training.

This module provides different reward configurations including sparse reward
mode for foundational learning.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RewardMode:
    """Reward mode configuration manager."""
    
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    
    def __init__(self, mode: str = HYBRID, config_path: Path = None):
        """Initialize reward mode.
        
        Args:
            mode: Reward mode ("sparse", "dense", or "hybrid")
            config_path: Path to rewards config file
        """
        self.mode = mode
        self.config_path = config_path or Path("configs/rewards.yaml")
        self.config = self._load_config()
        self.weights = self._get_weights()
        
        logger.info(f"Reward mode: {mode}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load reward configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_weights(self) -> Dict[str, float]:
        """Get reward weights based on mode.
        
        Returns:
            Dictionary of reward component weights
        """
        if self.mode == self.SPARSE:
            # Only sparse rewards (goals, saves, demos)
            return {
                "sparse": 1.0,
                "ball": 0.0,
                "goal": 0.0,
                "positioning": 0.0,
                "boost": 0.0,
                "aerial": 0.0,
                "mechanics": 0.0,
                "penalties": 0.0
            }
        elif self.mode == self.DENSE:
            # All dense rewards, minimal sparse
            return {
                "sparse": 0.2,
                "ball": 1.0,
                "goal": 1.0,
                "positioning": 1.0,
                "boost": 1.0,
                "aerial": 1.0,
                "mechanics": 1.0,
                "penalties": 1.0
            }
        else:  # HYBRID (default)
            # Balanced combination
            return {
                "sparse": 1.0,
                "ball": 0.5,
                "goal": 0.7,
                "positioning": 0.6,
                "boost": 0.5,
                "aerial": 0.8,
                "mechanics": 0.4,
                "penalties": 0.7
            }
    
    def scale_reward(self, reward_type: str, reward_value: float) -> float:
        """Scale a reward by the mode weight.
        
        Args:
            reward_type: Type of reward (e.g., "sparse", "ball", etc.)
            reward_value: Raw reward value
            
        Returns:
            Scaled reward value
        """
        weight = self.weights.get(reward_type, 1.0)
        return reward_value * weight
    
    def get_config(self) -> Dict[str, float]:
        """Get full reward configuration.
        
        Returns:
            Dictionary of all reward weights
        """
        config = {}
        for category, weight in self.weights.items():
            if category in self.config:
                for key, value in self.config[category].items():
                    config[f"{category}.{key}"] = value * weight
        return config
    
    @staticmethod
    def create_sparse_config(output_path: Path):
        """Create a sparse reward configuration file.
        
        Args:
            output_path: Path to save sparse config
        """
        sparse_config = {
            "sparse": {
                "goal_scored": 10.0,
                "goal_conceded": -10.0,
                "shot_on_goal": 2.0,
                "save": 3.0,
                "demo": 1.0,
                "demoed": -1.0
            },
            "ball": {},
            "goal": {},
            "positioning": {},
            "boost": {},
            "aerial": {},
            "mechanics": {},
            "penalties": {}
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sparse_config, f, default_flow_style=False)
        
        logger.info(f"Created sparse reward config at {output_path}")


def get_reward_mode_from_config(config: Dict[str, Any]) -> str:
    """Extract reward mode from training config.
    
    Args:
        config: Training configuration
        
    Returns:
        Reward mode string
    """
    return config.get("training", {}).get("reward_mode", RewardMode.HYBRID)
