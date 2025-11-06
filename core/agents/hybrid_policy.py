"""Hybrid policy combining rule-based and ML policies.

This module implements intelligent routing between rule-based tactics
and ML-driven decision making based on context and confidence.
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from .intents import Intent
from .rule_policy import RulePolicy, GameContext
from .ml_policy import MLPolicy
from core.features.encoder import ObservationEncoder, RawObservation


class HybridPolicy:
    """Hybrid policy that routes between rule-based and ML policies.
    
    Uses rules for:
    - Kickoffs
    - Low ML confidence situations
    - Out-of-distribution (OOD) scenarios
    - Actuator saturation detection
    
    Uses ML for general play when confident.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize hybrid policy.
        
        Args:
            model_path: Path to trained ML model
            config: Configuration dictionary with hybrid policy settings
        """
        self.config = config or {}
        hybrid_config = self.config.get("hybrid", {})
        
        # Routing settings
        self.use_rules_on_kickoff = hybrid_config.get("use_rules_on_kickoff", True)
        self.use_rules_on_low_confidence = hybrid_config.get("use_rules_on_low_confidence", True)
        self.confidence_threshold = hybrid_config.get("confidence_threshold", 0.7)
        self.ood_detection = hybrid_config.get("ood_detection", "entropy")
        self.ood_threshold = hybrid_config.get("ood_threshold", 2.0)
        self.fallback_on_saturation = hybrid_config.get("fallback_on_saturation", True)
        
        # Initialize sub-policies
        self.encoder = ObservationEncoder(config.get("encoder", {}))
        self.rule_policy = RulePolicy(self.config.get("rules", {}))
        self.ml_policy = MLPolicy(model_path, self.encoder, self.config)
        
        # State tracking
        self.last_controls = np.zeros(8)
        self.consecutive_saturations = 0
        self.saturation_threshold = 10  # Switch to rules after this many saturated actions
        
        # Statistics
        self.rule_activations = 0
        self.ml_activations = 0
        self.ood_detections = 0
        
        # LSTM hidden state (if using LSTM)
        self.hidden_state = None
    
    def get_action(
        self,
        raw_obs: RawObservation,
        game_context: GameContext
    ) -> Tuple[np.ndarray, Intent, float, str]:
        """Get action from hybrid policy.
        
        Args:
            raw_obs: Raw observation from the game
            game_context: Game context for rule policy
            
        Returns:
            Tuple of (controls, intent, confidence, source):
                - controls: 8-dimensional action array
                - intent: High-level intent
                - confidence: Confidence score (0-1)
                - source: "rule" or "ml" indicating which policy was used
        """
        # Encode observation for ML
        encoded_obs = self.encoder.encode(raw_obs)
        
        # Check routing conditions
        use_rules = False
        reason = ""
        
        # 1. Kickoff handling
        if self.use_rules_on_kickoff and game_context.is_kickoff:
            use_rules = True
            reason = "kickoff"
        
        # 2. ML confidence check (only if not already using rules)
        if not use_rules and self.use_rules_on_low_confidence:
            ml_confidence = self.ml_policy.get_confidence(encoded_obs)
            if ml_confidence < self.confidence_threshold:
                use_rules = True
                reason = "low_confidence"
        
        # 3. OOD detection
        if not use_rules and self._detect_ood(encoded_obs):
            use_rules = True
            reason = "ood"
            self.ood_detections += 1
        
        # 4. Actuator saturation detection
        if not use_rules and self.fallback_on_saturation:
            if self._detect_saturation(self.last_controls):
                use_rules = True
                reason = "saturation"
        
        # Get action from selected policy
        if use_rules:
            controls, intent, confidence = self.rule_policy.get_action(game_context)
            source = f"rule ({reason})"
            self.rule_activations += 1
            # Reset LSTM hidden state when switching to rules
            self.hidden_state = None
        else:
            controls, intent, confidence, self.hidden_state = self.ml_policy.get_action(
                encoded_obs,
                self.hidden_state,
                deterministic=False
            )
            source = "ml"
            self.ml_activations += 1
        
        # Track controls for saturation detection
        self.last_controls = controls
        
        return controls, intent, confidence, source
    
    def _detect_ood(self, observation: np.ndarray) -> bool:
        """Detect out-of-distribution observations.
        
        Args:
            observation: Encoded observation
            
        Returns:
            True if observation is OOD
        """
        if self.ood_detection == "entropy":
            # Use entropy of action distribution
            # Get confidence from ML policy (higher entropy = lower confidence)
            confidence = self.ml_policy.get_confidence(observation)
            return confidence < self.confidence_threshold
        
        elif self.ood_detection == "mahalanobis":
            # Mahalanobis distance (simplified - would need running statistics)
            # For now, use simple threshold on observation values
            obs_magnitude = np.linalg.norm(observation)
            return obs_magnitude > self.ood_threshold
        
        elif self.ood_detection == "autoencoder":
            # Autoencoder reconstruction error
            # TODO: Implement autoencoder-based OOD detection
            return False
        
        return False
    
    def _detect_saturation(self, controls: np.ndarray) -> bool:
        """Detect actuator saturation.
        
        If controls are at extreme values for many consecutive frames,
        the policy might be struggling.
        
        Args:
            controls: Control array
            
        Returns:
            True if saturation detected
        """
        # Check if any control is at extreme value
        saturated = np.any(np.abs(controls[:5]) > 0.95)  # Check continuous controls
        
        if saturated:
            self.consecutive_saturations += 1
        else:
            self.consecutive_saturations = 0
        
        return self.consecutive_saturations > self.saturation_threshold
    
    def reset(self):
        """Reset policy state."""
        self.encoder.reset()
        self.ml_policy.reset()
        self.hidden_state = None
        self.last_controls = np.zeros(8)
        self.consecutive_saturations = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about policy usage.
        
        Returns:
            Dictionary with statistics
        """
        total_activations = self.rule_activations + self.ml_activations
        
        stats = {
            "rule_activations": self.rule_activations,
            "ml_activations": self.ml_activations,
            "rule_percentage": (
                self.rule_activations / total_activations * 100
                if total_activations > 0 else 0
            ),
            "ml_percentage": (
                self.ml_activations / total_activations * 100
                if total_activations > 0 else 0
            ),
            "ood_detections": self.ood_detections,
        }
        
        # Add ML inference stats
        ml_stats = self.ml_policy.get_stats()
        stats.update(ml_stats)
        
        return stats
