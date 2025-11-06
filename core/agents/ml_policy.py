"""ML-based policy using PyTorch for inference.

This module handles neural network inference for decision-making,
with confidence estimation and fast CPU/GPU execution.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from .intents import Intent
from core.models.nets import ActorCriticNet
from core.features.encoder import ObservationEncoder


class MLPolicy:
    """ML-based policy for bot decision-making.
    
    Uses PyTorch neural network for inference with:
    - Fast evaluation (< 1ms target)
    - Confidence estimation
    - CPU/GPU support
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        encoder: Optional[ObservationEncoder] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ML policy.
        
        Args:
            model_path: Path to trained model checkpoint
            encoder: Observation encoder (if None, will create default)
            config: Configuration dictionary
        """
        self.config = config or {}
        self.device = torch.device(
            self.config.get("device", "cpu")
        )
        
        # Encoder
        self.encoder = encoder or ObservationEncoder()
        
        # Model
        self.model: Optional[ActorCriticNet] = None
        self.model_loaded = False
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            # Create default model architecture
            self._create_default_model()
        
        # Performance tracking
        self.inference_times = []
        self.max_inference_time = 0.0
    
    def _create_default_model(self):
        """Create a default model architecture."""
        network_config = self.config.get("network", {})
        
        self.model = ActorCriticNet(
            input_size=self.encoder.feature_size,
            hidden_sizes=network_config.get("hidden_sizes", [512, 512, 256]),
            action_categoricals=5,  # throttle, steer, pitch, yaw, roll
            action_bernoullis=3,    # jump, boost, handbrake
            activation=network_config.get("activation", "relu"),
            use_lstm=network_config.get("use_lstm", False),
            lstm_hidden_size=network_config.get("lstm_hidden_size", 256)
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path: Path):
        """Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model if not exists
            if self.model is None:
                self._create_default_model()
            
            # Load state dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_loaded = True
            
            print(f"MLPolicy: Loaded model from {model_path}")
        except Exception as e:
            print(f"MLPolicy: Failed to load model from {model_path}: {e}")
            self._create_default_model()
            self.model_loaded = False
    
    def get_action(
        self,
        observation: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Intent, float, Optional[Tuple]]:
        """Get action from ML policy.
        
        Args:
            observation: Encoded observation as numpy array
            hidden: Optional LSTM hidden state
            deterministic: If True, select argmax instead of sampling
            
        Returns:
            Tuple of (controls, intent, confidence, hidden):
                - controls: 8-dimensional action array
                - intent: High-level intent (estimated from actions)
                - confidence: Confidence score (0-1)
                - hidden: Updated hidden state (if using LSTM)
        """
        import time
        start_time = time.time()
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            cat_probs, ber_probs, value, confidence, hidden = self.model(
                obs_tensor,
                hidden
            )
        
        # Sample or select argmax
        if deterministic:
            actions_cat = torch.argmax(cat_probs, dim=2)
            actions_ber = torch.argmax(ber_probs, dim=2)
        else:
            # Sample from distributions
            actions_cat = torch.multinomial(
                cat_probs.view(-1, 3),
                1
            ).view(-1, self.model.action_categoricals)
            
            actions_ber = torch.multinomial(
                ber_probs.view(-1, 2),
                1
            ).view(-1, self.model.action_bernoullis)
        
        # Convert to numpy
        actions_cat = actions_cat.cpu().numpy()[0]  # (5,)
        actions_ber = actions_ber.cpu().numpy()[0]  # (3,)
        confidence_value = confidence.cpu().item()
        
        # Map to controls
        # Categorical: -1, 0, 1
        cat_mapping = np.array([-1.0, 0.0, 1.0])
        controls_cat = cat_mapping[actions_cat]
        
        # Bernoulli: 0, 1
        controls_ber = actions_ber.astype(np.float32)
        
        # Combine: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        controls = np.concatenate([controls_cat, controls_ber])
        
        # Estimate intent from controls
        intent = self._estimate_intent(controls)
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        self.max_inference_time = max(self.max_inference_time, inference_time)
        
        return controls, intent, confidence_value, hidden
    
    def _estimate_intent(self, controls: np.ndarray) -> Intent:
        """Estimate high-level intent from controls.
        
        This is a heuristic mapping from low-level controls to high-level intents.
        
        Args:
            controls: 8-dimensional control array
            
        Returns:
            Estimated intent
        """
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = controls
        
        # Simple heuristics
        if boost > 0 and abs(throttle) > 0.5:
            if throttle > 0:
                return Intent.CHALLENGE
            else:
                return Intent.ROTATE_BACK
        
        if jump > 0:
            return Intent.CHALLENGE  # Likely aerial or challenge
        
        if abs(throttle) > 0.5:
            if throttle > 0:
                return Intent.DRIVE_TO_BALL
            else:
                return Intent.ROTATE_BACK
        
        if handbrake > 0:
            return Intent.RECOVERY
        
        return Intent.DRIVE_TO_POSITION
    
    def get_value(self, observation: np.ndarray) -> float:
        """Get value estimate for observation.
        
        Args:
            observation: Encoded observation
            
        Returns:
            Value estimate
        """
        obs_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            value = self.model.get_value(obs_tensor)
        
        return value.cpu().item()
    
    def get_confidence(self, observation: np.ndarray) -> float:
        """Get confidence for observation.
        
        Uses entropy of action distribution as confidence measure.
        
        Args:
            observation: Encoded observation
            
        Returns:
            Confidence score (0-1)
        """
        obs_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            cat_probs, ber_probs, _, confidence, _ = self.model(obs_tensor)
        
        # Use model's confidence head
        return confidence.cpu().item()
    
    def reset(self):
        """Reset policy state (e.g., LSTM hidden state)."""
        if self.model and hasattr(self.model, "backbone"):
            if hasattr(self.model.backbone, "reset_hidden"):
                self.model.backbone.reset_hidden()
    
    def get_stats(self) -> Dict[str, float]:
        """Get inference statistics.
        
        Returns:
            Dictionary with inference timing stats
        """
        if not self.inference_times:
            return {
                "mean_inference_ms": 0.0,
                "max_inference_ms": 0.0,
                "min_inference_ms": 0.0
            }
        
        return {
            "mean_inference_ms": np.mean(self.inference_times),
            "max_inference_ms": self.max_inference_time,
            "min_inference_ms": np.min(self.inference_times)
        }
