"""Unit tests for rule policy."""
import numpy as np
import pytest

from core.agents.rule_policy import RulePolicy, GameContext
from core.agents.intents import Intent


def test_rule_policy_initialization():
    """Test rule policy initialization."""
    policy = RulePolicy()
    assert policy is not None
    assert policy.aggressive == False
    assert policy.boost_conservation == True


def test_rule_policy_kickoff():
    """Test kickoff action."""
    policy = RulePolicy()
    
    context = GameContext(
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
    
    controls, intent, confidence = policy.get_action(context)
    
    # Check output
    assert controls.shape == (8,)
    assert intent == Intent.KICKOFF
    assert confidence > 0.9  # High confidence for rules


def test_rule_policy_defensive():
    """Test defensive action."""
    policy = RulePolicy()
    
    context = GameContext(
        is_kickoff=False,
        ball_position=np.array([0.0, -4000.0, 500.0]),
        ball_velocity=np.array([0.0, -1000.0, 0.0]),
        car_position=np.array([0.0, -5000.0, 20.0]),
        car_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation=np.eye(3).flatten(),
        boost_amount=50.0,
        on_ground=True,
        has_flip=True,
        is_closest_to_ball=True,
        is_last_man=True,
        teammates_positions=[],
        opponents_positions=[],
        game_time=30.0,
        score_diff=0
    )
    
    controls, intent, confidence = policy.get_action(context)
    
    # Should choose defensive action
    assert intent == Intent.SAVE
    assert confidence > 0.9


def test_rule_policy_boost_pickup():
    """Test boost pickup when low."""
    policy = RulePolicy()
    
    context = GameContext(
        is_kickoff=False,
        ball_position=np.array([0.0, 1000.0, 100.0]),
        ball_velocity=np.array([0.0, 0.0, 0.0]),
        car_position=np.array([0.0, -1000.0, 20.0]),
        car_velocity=np.array([0.0, 0.0, 0.0]),
        car_rotation=np.eye(3).flatten(),
        boost_amount=5.0,  # Critical low
        on_ground=True,
        has_flip=True,
        is_closest_to_ball=False,
        is_last_man=False,
        teammates_positions=[],
        opponents_positions=[],
        game_time=30.0,
        score_diff=0
    )
    
    controls, intent, confidence = policy.get_action(context)
    
    # Should pick up boost
    assert intent == Intent.BOOST_PICKUP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
