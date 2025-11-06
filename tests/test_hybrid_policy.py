"""Unit tests for hybrid policy routing."""
import numpy as np
import pytest

from core.agents.hybrid_policy import HybridPolicy
from core.agents.rule_policy import GameContext
from core.features.encoder import RawObservation


def test_hybrid_policy_initialization():
    """Test hybrid policy initialization."""
    policy = HybridPolicy()
    assert policy is not None
    assert policy.rule_policy is not None
    assert policy.ml_policy is not None


def test_hybrid_policy_kickoff_routing():
    """Test that kickoff uses rule policy."""
    config = {"hybrid": {"use_rules_on_kickoff": True}}
    policy = HybridPolicy(config=config)
    
    # Create kickoff context
    raw_obs = RawObservation(
        car_position=np.array([0.0, -3000.0, 20.0]),
        car_velocity=np.array([0.0, 0.0, 0.0]),
        car_angular_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation_matrix=np.eye(3),
        car_boost=33.0,
        car_on_ground=True,
        car_has_flip=True,
        car_is_demoed=False,
        ball_position=np.array([0.0, 0.0, 100.0]),
        ball_velocity=np.array([0.0, 0.0, 0.0]),
        ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
        is_kickoff=True,
        game_time=0.0,
        score_self=0,
        score_opponent=0,
        game_phase="KICKOFF"
    )
    
    game_context = GameContext(
        is_kickoff=True,
        ball_position=np.array([0.0, 0.0, 100.0]),
        ball_velocity=np.array([0.0, 0.0, 0.0]),
        car_position=np.array([0.0, -3000.0, 20.0]),
        car_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation=np.eye(3).flatten(),
        boost_amount=33.0,
        on_ground=True,
        has_flip=True,
        is_closest_to_ball=True,
        is_last_man=False,
        teammates_positions=[],
        opponents_positions=[],
        game_time=0.0,
        score_diff=0
    )
    
    controls, intent, confidence, source = policy.get_action(raw_obs, game_context)
    
    # Should use rule policy for kickoff
    assert "rule" in source.lower()
    assert controls.shape == (8,)


def test_hybrid_policy_stats():
    """Test hybrid policy statistics."""
    policy = HybridPolicy()
    
    # Get stats
    stats = policy.get_stats()
    
    assert "rule_activations" in stats
    assert "ml_activations" in stats
    assert "rule_percentage" in stats


def test_hybrid_policy_reset():
    """Test hybrid policy reset."""
    policy = HybridPolicy()
    
    # Reset should not raise error
    policy.reset()
    
    # Check state is reset
    assert policy.hidden_state is None
    assert policy.consecutive_saturations == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
