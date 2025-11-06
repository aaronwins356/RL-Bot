"""Bot manager for loading and managing policies.

This module handles policy initialization and fallback logic.
"""
from pathlib import Path
from typing import Optional, Dict, Any

from core.agents.hybrid_policy import HybridPolicy
from core.agents.ml_policy import MLPolicy
from core.agents.rule_policy import RulePolicy
from core.infra.config import Config, load_config


class BotManager:
    """Manager for bot policies with fallback handling."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        policy_type: str = "hybrid"
    ):
        """Initialize bot manager.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model
            policy_type: Type of policy ("hybrid", "ml", or "rule")
        """
        # Load config
        if config_path and config_path.exists():
            self.config = load_config(config_path)
        else:
            # Default config
            self.config = Config()
        
        self.model_path = model_path
        self.policy_type = policy_type or self.config.policy_type
        
        # Initialize policy
        self.policy = self._create_policy()
    
    def _create_policy(self):
        """Create policy based on configuration.
        
        Returns:
            Policy instance
        """
        try:
            if self.policy_type == "hybrid":
                policy = HybridPolicy(
                    model_path=self.model_path,
                    config=self.config.raw_config
                )
                print(f"BotManager: Loaded hybrid policy")
                return policy
            
            elif self.policy_type == "ml":
                policy = MLPolicy(
                    model_path=self.model_path,
                    config=self.config.raw_config
                )
                print(f"BotManager: Loaded ML policy")
                return policy
            
            elif self.policy_type == "rule":
                policy = RulePolicy(
                    config=self.config.raw_config.get("rules", {})
                )
                print(f"BotManager: Loaded rule-based policy")
                return policy
            
            else:
                raise ValueError(f"Unknown policy type: {self.policy_type}")
        
        except Exception as e:
            print(f"BotManager: Failed to load {self.policy_type} policy: {e}")
            print(f"BotManager: Falling back to rule-based policy")
            return RulePolicy()
    
    def get_policy(self):
        """Get the active policy.
        
        Returns:
            Policy instance
        """
        return self.policy
    
    def reload_policy(self, model_path: Optional[Path] = None):
        """Reload policy (hot reload).
        
        Args:
            model_path: Optional new model path
        """
        if model_path:
            self.model_path = model_path
        
        print(f"BotManager: Reloading policy...")
        self.policy = self._create_policy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "policy_type": self.policy_type,
            "model_path": str(self.model_path) if self.model_path else None
        }
        
        # Add policy-specific stats
        if hasattr(self.policy, "get_stats"):
            policy_stats = self.policy.get_stats()
            stats.update(policy_stats)
        
        return stats
