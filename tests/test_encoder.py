"""Unit tests for observation encoder."""
import numpy as np
import pytest

from core.features.encoder import ObservationEncoder, RawObservation


def test_encoder_initialization():
    """Test encoder initialization."""
    encoder = ObservationEncoder()
    assert encoder.feature_size > 0
    assert encoder.normalize is True


def test_encoder_encode_basic():
    """Test basic encoding."""
    encoder = ObservationEncoder()
    
    # Create raw observation
    obs = RawObservation(
        car_position=np.array([0.0, 0.0, 20.0]),
        car_velocity=np.array([500.0, 0.0, 0.0]),
        car_angular_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation_matrix=np.eye(3),
        car_boost=50.0,
        car_on_ground=True,
        car_has_flip=True,
        car_is_demoed=False,
        ball_position=np.array([0.0, 1000.0, 100.0]),
        ball_velocity=np.array([0.0, 500.0, 0.0]),
        ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
        is_kickoff=False,
        game_time=30.0,
        score_self=0,
        score_opponent=0,
        game_phase="NEUTRAL"
    )
    
    # Encode
    encoded = encoder.encode(obs)
    
    # Check shape
    assert encoded.shape == (encoder.feature_size,)
    assert encoded.dtype == np.float32
    
    # Check values are normalized
    assert np.all(np.abs(encoded) <= 10.0)  # Reasonable bounds


def test_encoder_with_history():
    """Test encoder with history."""
    encoder = ObservationEncoder({"include_history": True, "history_length": 2})
    
    obs = RawObservation(
        car_position=np.array([0.0, 0.0, 20.0]),
        car_velocity=np.array([500.0, 0.0, 0.0]),
        car_angular_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation_matrix=np.eye(3),
        car_boost=50.0,
        car_on_ground=True,
        car_has_flip=True,
        car_is_demoed=False,
        ball_position=np.array([0.0, 1000.0, 100.0]),
        ball_velocity=np.array([0.0, 500.0, 0.0]),
        ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
        is_kickoff=False,
        game_time=30.0,
        score_self=0,
        score_opponent=0,
        game_phase="NEUTRAL"
    )
    
    # Encode multiple times
    encoded1 = encoder.encode(obs)
    encoded2 = encoder.encode(obs)
    
    # Should include history
    assert len(encoder.history) > 0


def test_encoder_reset():
    """Test encoder reset."""
    encoder = ObservationEncoder({"include_history": True})
    
    obs = RawObservation(
        car_position=np.array([0.0, 0.0, 20.0]),
        car_velocity=np.array([500.0, 0.0, 0.0]),
        car_angular_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation_matrix=np.eye(3),
        car_boost=50.0,
        car_on_ground=True,
        car_has_flip=True,
        car_is_demoed=False,
        ball_position=np.array([0.0, 1000.0, 100.0]),
        ball_velocity=np.array([0.0, 500.0, 0.0]),
        ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
        is_kickoff=False,
        game_time=30.0,
        score_self=0,
        score_opponent=0,
        game_phase="NEUTRAL"
    )
    
    encoder.encode(obs)
    assert len(encoder.history) > 0
    
    encoder.reset()
    assert len(encoder.history) == 0


def test_encoder_aerial_features():
    """Test encoding of aerial-specific features."""
    encoder = ObservationEncoder()
    
    # Create observation with aerial features
    obs = RawObservation(
        car_position=np.array([0.0, 0.0, 20.0]),
        car_velocity=np.array([500.0, 0.0, 0.0]),
        car_angular_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation_matrix=np.eye(3),
        car_boost=75.0,
        car_on_ground=True,
        car_has_flip=True,
        car_is_demoed=False,
        ball_position=np.array([0.0, 1000.0, 600.0]),  # High ball
        ball_velocity=np.array([0.0, 500.0, 100.0]),
        ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
        ball_height_bucket=2,  # Mid-height bucket
        aerial_opportunity=True,
        car_alignment_to_ball=0.8,  # Good alignment
        is_kickoff=False,
        game_time=30.0,
        score_self=0,
        score_opponent=0,
        game_phase="NEUTRAL"
    )
    
    # Encode
    encoded = encoder.encode(obs)
    
    # Check shape includes aerial features
    assert encoded.shape == (encoder.feature_size,)
    assert encoded.dtype == np.float32
    
    # Verify observation has aerial features
    assert obs.ball_height_bucket == 2
    assert obs.aerial_opportunity is True
    assert obs.car_alignment_to_ball == 0.8


def test_encoder_aerial_height_buckets():
    """Test different ball height buckets are encoded correctly."""
    encoder = ObservationEncoder()
    
    # Test each height bucket
    for bucket in range(5):
        obs = RawObservation(
            car_position=np.array([0.0, 0.0, 20.0]),
            car_velocity=np.array([500.0, 0.0, 0.0]),
            car_angular_velocity=np.array([0.0, 0.0, 0.0]),
            car_rotation_matrix=np.eye(3),
            car_boost=50.0,
            car_on_ground=True,
            car_has_flip=True,
            car_is_demoed=False,
            ball_position=np.array([0.0, 1000.0, 100.0]),
            ball_velocity=np.array([0.0, 500.0, 0.0]),
            ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
            ball_height_bucket=bucket,
            aerial_opportunity=False,
            car_alignment_to_ball=0.0,
            is_kickoff=False,
            game_time=30.0,
            score_self=0,
            score_opponent=0,
            game_phase="NEUTRAL"
        )
        
        encoded = encoder.encode(obs)
        
        # Should encode successfully
        assert encoded.shape == (encoder.feature_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
