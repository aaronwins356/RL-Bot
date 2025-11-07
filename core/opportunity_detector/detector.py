"""Opportunity Detector for classifying game state and selecting Skill Programs."""

from typing import Dict, Any, Tuple, Optional
from enum import Enum
import numpy as np
import torch
import torch.nn as nn


class GameStateCategory(Enum):
    """Game state categories."""
    GROUND_PLAY = 0
    WALL_PLAY = 1
    AERIAL_OPPORTUNITY = 2
    FLASHY_OPPORTUNITY = 3


class OpportunityDetectorModel(nn.Module):
    """Neural network for opportunity detection.
    
    Small transformer or 2-layer Bi-LSTM (hidden 256), 0.5s context window.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OD model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.obs_dim = config.get('obs_dim', 180)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_categories = len(GameStateCategory)
        self.use_lstm = config.get('use_lstm', True)
        
        if self.use_lstm:
            # Bi-LSTM model
            self.lstm = nn.LSTM(
                input_size=self.obs_dim,
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.1,
            )
            lstm_output_dim = self.hidden_dim * 2  # Bidirectional
        else:
            # Transformer model
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.obs_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            lstm_output_dim = self.obs_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_categories),
        )
        
    def forward(self, obs_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs_sequence: [batch, sequence_len, obs_dim]
            
        Returns:
            Logits [batch, num_categories]
        """
        if self.use_lstm:
            # LSTM forward
            lstm_out, _ = self.lstm(obs_sequence)
            # Use last timestep output
            features = lstm_out[:, -1, :]
        else:
            # Transformer forward
            trans_out = self.transformer(obs_sequence)
            # Use last timestep output
            features = trans_out[:, -1, :]
        
        logits = self.classifier(features)
        return logits


class OpportunityDetector:
    """Opportunity Detector that classifies game state and selects Skill Programs.
    
    Classifies into {Ground Play, Wall Play, Aerial Opportunity, Flashy Opportunity}
    and selects a Skill Program (SP) with temperature-controlled sampling.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """Initialize Opportunity Detector.
        
        Args:
            config: Configuration dict
            device: Device for inference
        """
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = OpportunityDetectorModel(config.get('model', {}))
        self.model.to(device)
        self.model.eval()
        
        # Temperature for sampling
        self.temperature = config.get('temperature', 1.0)
        self.safe_option_floor = config.get('safe_option_floor', 0.2)
        
        # Context window (0.5s at 120Hz = 60 frames)
        self.context_window_frames = config.get('context_window_frames', 60)
        self.observation_buffer = []
        
        # SP mapping
        self.category_to_sp_map = {
            GameStateCategory.GROUND_PLAY: ['SP_AerialControl'],
            GameStateCategory.WALL_PLAY: ['SP_WallRead', 'SP_BackboardRead'],
            GameStateCategory.AERIAL_OPPORTUNITY: ['SP_FastAerial', 'SP_AerialControl'],
            GameStateCategory.FLASHY_OPPORTUNITY: [
                'SP_CeilingShot', 'SP_FlipReset', 'SP_Musty', 
                'SP_Breezi', 'SP_DoubleTap', 'SP_GroundToAirDribble'
            ],
        }
        
        # Bandit state for SP selection (Thompson sampling)
        self.sp_alpha = {}  # Success counts
        self.sp_beta = {}   # Failure counts
        for sps in self.category_to_sp_map.values():
            for sp in sps:
                self.sp_alpha[sp] = 1.0
                self.sp_beta[sp] = 1.0
        
    def reset(self):
        """Reset detector state."""
        self.observation_buffer = []
        
    def update_bandit(self, sp_name: str, success: bool):
        """Update bandit statistics.
        
        Args:
            sp_name: Name of skill program
            success: Whether SP was successful
        """
        if sp_name in self.sp_alpha:
            if success:
                self.sp_alpha[sp_name] += 1.0
            else:
                self.sp_beta[sp_name] += 1.0
    
    def detect(self, obs: Dict[str, Any], risk_score: float) -> Tuple[GameStateCategory, str, float]:
        """Detect game state category and select SP.
        
        Args:
            obs: Current observation
            risk_score: Risk score from RiskScorer (0-1)
            
        Returns:
            Tuple of (category, sp_name, confidence)
        """
        # Add observation to buffer
        obs_vector = self._extract_obs_vector(obs)
        self.observation_buffer.append(obs_vector)
        
        # Keep only context window
        if len(self.observation_buffer) > self.context_window_frames:
            self.observation_buffer = self.observation_buffer[-self.context_window_frames:]
        
        # Need enough history
        if len(self.observation_buffer) < 10:
            return GameStateCategory.GROUND_PLAY, 'SP_AerialControl', 0.5
        
        # Prepare input sequence
        obs_sequence = np.array(self.observation_buffer)
        obs_tensor = torch.FloatTensor(obs_sequence).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = torch.softmax(logits / self.temperature, dim=-1)
        
        probs_np = probs.cpu().numpy()[0]
        
        # Apply safe option floor when risk is high
        if risk_score > 0.7:
            # Reduce flashy opportunity probability
            probs_np[GameStateCategory.FLASHY_OPPORTUNITY.value] *= 0.3
            # Renormalize
            probs_np = probs_np / probs_np.sum()
        
        # Sample category
        category_idx = np.random.choice(len(probs_np), p=probs_np)
        category = GameStateCategory(category_idx)
        confidence = float(probs_np[category_idx])
        
        # Select SP within category using Thompson sampling
        sp_name = self._select_sp(category)
        
        return category, sp_name, confidence
    
    def _extract_obs_vector(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract observation vector from obs dict.
        
        Args:
            obs: Observation dict
            
        Returns:
            Observation vector
        """
        # TODO: This should match the encoder output format from core/features/encoder.py
        # For now, create a placeholder that extracts basic features
        obs_dim = self.config.get('model', {}).get('obs_dim', 180)
        
        # Extract basic features if available
        features = []
        
        # Car state (position, velocity, orientation)
        car_pos = obs.get('car_position', np.zeros(3))
        car_vel = obs.get('car_velocity', np.zeros(3))
        features.extend(car_pos)
        features.extend(car_vel)
        
        # Ball state (position, velocity)
        ball_pos = obs.get('ball_position', np.zeros(3))
        ball_vel = obs.get('ball_velocity', np.zeros(3))
        features.extend(ball_pos)
        features.extend(ball_vel)
        
        # Pad to correct dimension
        feature_array = np.array(features)
        if len(feature_array) < obs_dim:
            feature_array = np.pad(feature_array, (0, obs_dim - len(feature_array)), 'constant')
        elif len(feature_array) > obs_dim:
            feature_array = feature_array[:obs_dim]
        
        return feature_array
    
    def _select_sp(self, category: GameStateCategory) -> str:
        """Select SP within category using Thompson sampling.
        
        Args:
            category: Game state category
            
        Returns:
            SP name
        """
        candidate_sps = self.category_to_sp_map[category]
        
        if len(candidate_sps) == 1:
            return candidate_sps[0]
        
        # Thompson sampling: sample from Beta distribution
        samples = []
        for sp in candidate_sps:
            alpha = self.sp_alpha.get(sp, 1.0)
            beta = self.sp_beta.get(sp, 1.0)
            sample = np.random.beta(alpha, beta)
            samples.append((sample, sp))
        
        # Select SP with highest sample
        samples.sort(reverse=True)
        return samples[0][1]
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'bandit_state' in checkpoint:
            self.sp_alpha = checkpoint['bandit_state']['alpha']
            self.sp_beta = checkpoint['bandit_state']['beta']
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'bandit_state': {
                'alpha': self.sp_alpha,
                'beta': self.sp_beta,
            },
        }
        torch.save(checkpoint, path)
