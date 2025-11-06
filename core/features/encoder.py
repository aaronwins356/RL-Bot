"""Observation encoder for converting game state to feature vectors.

This module handles encoding of Rocket League game state into normalized
feature vectors suitable for neural network input.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class RawObservation:
    """Raw observation from the game."""
    # Car state
    car_position: np.ndarray  # (3,) - x, y, z
    car_velocity: np.ndarray  # (3,)
    car_angular_velocity: np.ndarray  # (3,)
    car_rotation_matrix: np.ndarray  # (3, 3) - forward, right, up
    car_boost: float  # 0-100
    car_on_ground: bool
    car_has_flip: bool
    car_is_demoed: bool
    
    # Ball state
    ball_position: np.ndarray  # (3,)
    ball_velocity: np.ndarray  # (3,)
    ball_angular_velocity: np.ndarray  # (3,)
    
    # Ball prediction (optional)
    ball_predicted_position: Optional[np.ndarray] = None  # (3,) - position at intercept
    ball_predicted_time: Optional[float] = None  # Time to intercept
    
    # Aerial-specific features
    ball_height_bucket: Optional[int] = None  # 0=ground, 1=low, 2=mid, 3=high (0-4)
    aerial_opportunity: bool = False  # Flag indicating aerial opportunity
    car_alignment_to_ball: float = 0.0  # Dot product of car forward to ball direction
    
    # Teammates (list of dicts with position, velocity, boost, etc.)
    teammates: Optional[List[Dict[str, Any]]] = None
    
    # Opponents (list of dicts)
    opponents: Optional[List[Dict[str, Any]]] = None
    
    # Boost pads
    boost_pads: Optional[List[Dict[str, Any]]] = None  # List of {position, active, is_large}
    
    # Game state
    is_kickoff: bool = False
    game_time: float = 0.0
    score_self: int = 0
    score_opponent: int = 0
    
    # Context
    rotation_index: int = 0  # Position in rotation (0=attacker, 1=support, 2=defense)
    game_phase: str = "NEUTRAL"  # KICKOFF, OFFENSE, DEFENSE, NEUTRAL


class ObservationEncoder:
    """Encodes game observations into feature vectors for neural networks.
    
    The encoder normalizes and structures observations to create a consistent
    input format for ML models.
    """
    
    # Normalization constants
    POS_NORM = 4096.0  # Field is ~8192 x 10240
    VEL_NORM = 2300.0  # Max car speed
    ANG_VEL_NORM = 5.5  # Max angular velocity
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize encoder.
        
        Args:
            config: Configuration with settings like:
                - normalize: bool, whether to normalize values
                - include_history: bool, include previous observations
                - history_length: int, number of previous frames to include
        """
        self.config = config or {}
        self.normalize = self.config.get("normalize", True)
        self.include_history = self.config.get("include_history", False)
        self.history_length = self.config.get("history_length", 4)
        
        # History buffer
        self.history: List[np.ndarray] = []
        
        # Calculate feature size
        self.feature_size = self._calculate_feature_size()
    
    def _calculate_feature_size(self) -> int:
        """Calculate the total size of the feature vector."""
        # Car state: position(3) + velocity(3) + ang_vel(3) + rotation(9) + boost(1) + flags(3) = 22
        # Ball state: position(3) + velocity(3) + ang_vel(3) = 9
        # Ball relative: rel_pos(3) + rel_vel(3) = 6
        # Ball prediction: pred_pos(3) + pred_time(1) = 4
        # Aerial features: height_bucket_onehot(5) + aerial_opp(1) + alignment(1) = 7
        # Teammate (per): position(3) + velocity(3) + rel_pos(3) + rel_vel(3) + boost(1) = 13
        # Opponent (per): same as teammate = 13
        # Boost pads (per): position(3) + active(1) + is_large(1) + distance(1) = 6
        # Game state: is_kickoff(1) + time(1) + score_diff(1) = 3
        # Phase encoding: one-hot(4) = 4
        
        base_size = 22 + 9 + 6 + 4 + 7 + 3 + 4  # = 55
        
        # Assume max 2 teammates, 3 opponents, 34 boost pads
        max_teammates = 2
        max_opponents = 3
        max_boost_pads = 10  # Only encode nearest 10 boost pads
        
        total = base_size + (max_teammates * 13) + (max_opponents * 13) + (max_boost_pads * 6)
        # = 55 + 26 + 39 + 60 = 180
        
        if self.include_history:
            total *= (1 + self.history_length)
        
        return total
    
    def reset(self):
        """Reset the encoder state (clear history)."""
        self.history.clear()
    
    def encode(self, obs: RawObservation) -> np.ndarray:
        """Encode observation into feature vector.
        
        Args:
            obs: Raw observation from the game
            
        Returns:
            Normalized feature vector as numpy array
        """
        features = []
        
        # Car state
        features.extend(self._encode_car(obs))
        
        # Ball state
        features.extend(self._encode_ball(obs))
        
        # Ball relative to car
        features.extend(self._encode_ball_relative(obs))
        
        # Ball prediction
        features.extend(self._encode_ball_prediction(obs))
        
        # Aerial features
        features.extend(self._encode_aerial_features(obs))
        
        # Teammates
        features.extend(self._encode_teammates(obs))
        
        # Opponents
        features.extend(self._encode_opponents(obs))
        
        # Boost pads
        features.extend(self._encode_boost_pads(obs))
        
        # Game state
        features.extend(self._encode_game_state(obs))
        
        # Phase encoding
        features.extend(self._encode_phase(obs))
        
        feature_array = np.array(features, dtype=np.float32)
        
        # Add history if enabled
        if self.include_history:
            self.history.append(feature_array)
            if len(self.history) > self.history_length:
                self.history.pop(0)
            
            # Pad history if not enough frames yet
            while len(self.history) < self.history_length:
                self.history.insert(0, np.zeros_like(feature_array))
            
            # Concatenate current + history
            feature_array = np.concatenate([feature_array] + self.history)
        
        return feature_array
    
    def _encode_car(self, obs: RawObservation) -> List[float]:
        """Encode car state."""
        features = []
        
        # Position
        pos = obs.car_position / self.POS_NORM if self.normalize else obs.car_position
        features.extend(pos.tolist())
        
        # Velocity
        vel = obs.car_velocity / self.VEL_NORM if self.normalize else obs.car_velocity
        features.extend(vel.tolist())
        
        # Angular velocity
        ang_vel = obs.car_angular_velocity / self.ANG_VEL_NORM if self.normalize else obs.car_angular_velocity
        features.extend(ang_vel.tolist())
        
        # Rotation matrix (flattened)
        features.extend(obs.car_rotation_matrix.flatten().tolist())
        
        # Boost (normalized to 0-1)
        features.append(obs.car_boost / 100.0)
        
        # Flags
        features.append(float(obs.car_on_ground))
        features.append(float(obs.car_has_flip))
        features.append(float(obs.car_is_demoed))
        
        return features
    
    def _encode_ball(self, obs: RawObservation) -> List[float]:
        """Encode ball state."""
        features = []
        
        # Position
        pos = obs.ball_position / self.POS_NORM if self.normalize else obs.ball_position
        features.extend(pos.tolist())
        
        # Velocity
        vel = obs.ball_velocity / self.VEL_NORM if self.normalize else obs.ball_velocity
        features.extend(vel.tolist())
        
        # Angular velocity
        ang_vel = obs.ball_angular_velocity / self.ANG_VEL_NORM if self.normalize else obs.ball_angular_velocity
        features.extend(ang_vel.tolist())
        
        return features
    
    def _encode_ball_relative(self, obs: RawObservation) -> List[float]:
        """Encode ball position/velocity relative to car."""
        features = []
        
        # Relative position
        rel_pos = obs.ball_position - obs.car_position
        if self.normalize:
            rel_pos = rel_pos / self.POS_NORM
        features.extend(rel_pos.tolist())
        
        # Relative velocity
        rel_vel = obs.ball_velocity - obs.car_velocity
        if self.normalize:
            rel_vel = rel_vel / self.VEL_NORM
        features.extend(rel_vel.tolist())
        
        return features
    
    def _encode_ball_prediction(self, obs: RawObservation) -> List[float]:
        """Encode ball prediction."""
        features = []
        
        if obs.ball_predicted_position is not None:
            pred_pos = obs.ball_predicted_position / self.POS_NORM if self.normalize else obs.ball_predicted_position
            features.extend(pred_pos.tolist())
            features.append(obs.ball_predicted_time / 4.0 if obs.ball_predicted_time is not None else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _encode_aerial_features(self, obs: RawObservation) -> List[float]:
        """Encode aerial-specific features.
        
        Includes:
        - Ball height bucket (one-hot encoded)
        - Aerial opportunity flag
        - Car alignment to ball
        
        Args:
            obs: Raw observation
            
        Returns:
            List of aerial features
        """
        features = []
        
        # Ball height bucket (one-hot: ground, low, mid, high, very_high)
        # Buckets: 0-200, 200-500, 500-1000, 1000-1500, 1500+
        height_bucket = obs.ball_height_bucket if obs.ball_height_bucket is not None else 0
        height_onehot = [0.0] * 5
        if 0 <= height_bucket < 5:
            height_onehot[height_bucket] = 1.0
        features.extend(height_onehot)
        
        # Aerial opportunity flag
        features.append(float(obs.aerial_opportunity))
        
        # Car alignment to ball (dot product of forward vector to ball direction)
        # Already in range [-1, 1]
        features.append(obs.car_alignment_to_ball)
        
        return features
    
    def _encode_teammates(self, obs: RawObservation) -> List[float]:
        """Encode teammates information."""
        features = []
        max_teammates = 2
        
        teammates = obs.teammates or []
        
        for i in range(max_teammates):
            if i < len(teammates):
                tm = teammates[i]
                # Position
                pos = np.array(tm['position']) / self.POS_NORM if self.normalize else np.array(tm['position'])
                features.extend(pos.tolist())
                # Velocity
                vel = np.array(tm['velocity']) / self.VEL_NORM if self.normalize else np.array(tm['velocity'])
                features.extend(vel.tolist())
                # Relative position
                rel_pos = (np.array(tm['position']) - obs.car_position)
                if self.normalize:
                    rel_pos = rel_pos / self.POS_NORM
                features.extend(rel_pos.tolist())
                # Relative velocity
                rel_vel = (np.array(tm['velocity']) - obs.car_velocity)
                if self.normalize:
                    rel_vel = rel_vel / self.VEL_NORM
                features.extend(rel_vel.tolist())
                # Boost
                features.append(tm.get('boost', 0.0) / 100.0)
            else:
                # Padding
                features.extend([0.0] * 13)
        
        return features
    
    def _encode_opponents(self, obs: RawObservation) -> List[float]:
        """Encode opponents information."""
        features = []
        max_opponents = 3
        
        opponents = obs.opponents or []
        
        for i in range(max_opponents):
            if i < len(opponents):
                opp = opponents[i]
                # Position
                pos = np.array(opp['position']) / self.POS_NORM if self.normalize else np.array(opp['position'])
                features.extend(pos.tolist())
                # Velocity
                vel = np.array(opp['velocity']) / self.VEL_NORM if self.normalize else np.array(opp['velocity'])
                features.extend(vel.tolist())
                # Relative position
                rel_pos = (np.array(opp['position']) - obs.car_position)
                if self.normalize:
                    rel_pos = rel_pos / self.POS_NORM
                features.extend(rel_pos.tolist())
                # Relative velocity
                rel_vel = (np.array(opp['velocity']) - obs.car_velocity)
                if self.normalize:
                    rel_vel = rel_vel / self.VEL_NORM
                features.extend(rel_vel.tolist())
                # Boost
                features.append(opp.get('boost', 0.0) / 100.0)
            else:
                # Padding
                features.extend([0.0] * 13)
        
        return features
    
    def _encode_boost_pads(self, obs: RawObservation) -> List[float]:
        """Encode nearest boost pads."""
        features = []
        max_pads = 10
        
        boost_pads = obs.boost_pads or []
        
        # Sort by distance to car
        if boost_pads:
            pads_with_dist = []
            for pad in boost_pads:
                dist = np.linalg.norm(np.array(pad['position']) - obs.car_position)
                pads_with_dist.append((dist, pad))
            pads_with_dist.sort(key=lambda x: x[0])
            boost_pads = [pad for _, pad in pads_with_dist]
        
        for i in range(max_pads):
            if i < len(boost_pads):
                pad = boost_pads[i]
                # Position
                pos = np.array(pad['position']) / self.POS_NORM if self.normalize else np.array(pad['position'])
                features.extend(pos.tolist())
                # Active
                features.append(float(pad.get('active', True)))
                # Is large
                features.append(float(pad.get('is_large', False)))
                # Distance
                dist = np.linalg.norm(np.array(pad['position']) - obs.car_position)
                features.append(dist / self.POS_NORM if self.normalize else dist)
            else:
                # Padding
                features.extend([0.0] * 6)
        
        return features
    
    def _encode_game_state(self, obs: RawObservation) -> List[float]:
        """Encode game state."""
        features = []
        
        # Is kickoff
        features.append(float(obs.is_kickoff))
        
        # Game time (normalized to 0-1, assuming 5 min match)
        features.append(obs.game_time / 300.0 if self.normalize else obs.game_time)
        
        # Score difference (normalized to -1 to 1)
        score_diff = obs.score_self - obs.score_opponent
        features.append(np.tanh(score_diff / 5.0) if self.normalize else score_diff)
        
        return features
    
    def _encode_phase(self, obs: RawObservation) -> List[float]:
        """Encode game phase as one-hot."""
        phases = ["KICKOFF", "OFFENSE", "DEFENSE", "NEUTRAL"]
        one_hot = [0.0] * len(phases)
        
        if obs.game_phase in phases:
            idx = phases.index(obs.game_phase)
            one_hot[idx] = 1.0
        else:
            one_hot[-1] = 1.0  # Default to NEUTRAL
        
        return one_hot
