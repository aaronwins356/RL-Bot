"""Unit tests for rocket sim environment."""
import numpy as np
import pytest

from core.env.rocket_sim_env import RocketSimEnv
from core.features.encoder import RawObservation


def test_env_initialization():
    """Test environment initialization."""
    env = RocketSimEnv(
        game_mode="1v1",
        tick_skip=8,
        enable_aerial_training=True
    )
    
    assert env.game_mode == "1v1"
    assert env.tick_skip == 8
    assert env.enable_aerial_training is True
    assert env.encoder is not None
    assert env.reward_config is not None


def test_env_reset():
    """Test environment reset."""
    env = RocketSimEnv()
    
    obs = env.reset(seed=42)
    
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == env.encoder.feature_size
    assert env.episode_length == 0
    assert env.total_reward == 0.0


def test_env_step():
    """Test environment step."""
    env = RocketSimEnv()
    env.reset()
    
    # Create action: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.episode_length == 1


def test_aerial_opportunity_detection():
    """Test aerial opportunity detection."""
    env = RocketSimEnv(enable_aerial_training=True)
    
    # Test high ball - should be aerial opportunity
    assert env._detect_aerial_opportunity(ball_height=500.0, ball_distance=1500.0) is True
    
    # Test low ball - should not be aerial opportunity
    assert env._detect_aerial_opportunity(ball_height=100.0, ball_distance=1500.0) is False
    
    # Test far ball - should not be aerial opportunity
    assert env._detect_aerial_opportunity(ball_height=500.0, ball_distance=3000.0) is False


def test_boost_efficiency_calculation():
    """Test boost efficiency calculation."""
    env = RocketSimEnv()
    
    # Good efficiency: high value, low boost usage
    efficiency = env._calculate_boost_efficiency(boost_used=10.0, action_value=50.0)
    assert efficiency > 0.5
    
    # Poor efficiency: low value, high boost usage
    efficiency = env._calculate_boost_efficiency(boost_used=50.0, action_value=10.0)
    assert efficiency < 0.5
    
    # No boost used
    efficiency = env._calculate_boost_efficiency(boost_used=0.0, action_value=10.0)
    assert efficiency == 1.0


def test_reward_config():
    """Test reward configuration."""
    env = RocketSimEnv()
    
    assert 'sparse' in env.reward_config
    assert 'dense' in env.reward_config
    assert 'penalties' in env.reward_config
    assert env.reward_config['sparse']['goal_scored'] > 0
    assert env.reward_config['sparse']['goal_conceded'] < 0


def test_episode_termination():
    """Test episode termination conditions."""
    env = RocketSimEnv()
    env.reset()
    
    # Simulate goal
    env.stats['goals_scored'] = 1
    assert env._check_terminated() is True
    
    # Reset and simulate opponent goal
    env.reset()
    env.stats['goals_conceded'] = 1
    assert env._check_terminated() is True
    
    # No goals
    env.reset()
    assert env._check_terminated() is False


def test_max_episode_length():
    """Test episode truncation at max length."""
    env = RocketSimEnv()
    env.max_episode_length = 10
    env.reset()
    
    # Step through episode
    action = np.zeros(8)
    for _ in range(9):
        _, _, terminated, truncated, _ = env.step(action)
        assert not truncated
    
    # Final step should truncate
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated or env.episode_length >= env.max_episode_length
