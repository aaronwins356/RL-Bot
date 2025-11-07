"""Unit tests for environment wrappers."""
import numpy as np
import pytest

from core.env.rocket_sim_env import RocketSimEnv
from core.env.wrappers import (
    NormalizeObservation,
    FrameStack,
    RewardShaping,
    AerialTrainingWrapper,
    BoostManagementWrapper
)


def test_normalize_observation_wrapper():
    """Test observation normalization wrapper."""
    env = RocketSimEnv()
    wrapped_env = NormalizeObservation(env)
    
    result = wrapped_env.reset()
    # Handle new Gym API
    if isinstance(result, tuple):
        obs, _ = result
    else:
        obs = result
    
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    
    # Step and check normalization
    action = np.zeros(8)
    obs, _, _, _, _ = wrapped_env.step(action)
    
    # Normalized obs should be reasonable
    assert np.all(np.abs(obs) < 20.0)  # Clipped to reasonable range


def test_frame_stack_wrapper():
    """Test frame stacking wrapper."""
    env = RocketSimEnv()
    wrapped_env = FrameStack(env, num_stack=4)
    
    result = wrapped_env.reset()
    # Handle new Gym API
    if isinstance(result, tuple):
        obs, _ = result
    else:
        obs = result
    
    # Stacked obs should be 4x the size
    base_size = env.encoder.feature_size
    assert obs.shape[0] == base_size * 4
    
    # Step and verify stack updates
    action = np.zeros(8)
    obs_after, _, _, _, _ = wrapped_env.step(action)
    assert obs_after.shape[0] == base_size * 4


def test_reward_shaping_wrapper():
    """Test reward shaping wrapper."""
    env = RocketSimEnv()
    
    # Test with reward scale
    wrapped_env = RewardShaping(env, reward_scale=2.0)
    wrapped_env.reset()
    
    action = np.zeros(8)
    _, reward, _, _, _ = wrapped_env.step(action)
    
    # Reward should be scaled (though may be 0 in base implementation)
    assert isinstance(reward, float)
    
    # Test with reward clipping
    wrapped_env = RewardShaping(env, reward_clip=(-1.0, 1.0))
    wrapped_env.reset()
    _, reward, _, _, _ = wrapped_env.step(action)
    assert -1.0 <= reward <= 1.0


def test_reward_shaping_custom_function():
    """Test reward shaping with custom function."""
    env = RocketSimEnv()
    
    # Custom reward function that doubles the reward
    def custom_reward_fn(reward, info):
        return reward * 2.0
    
    wrapped_env = RewardShaping(env, reward_fn=custom_reward_fn)
    wrapped_env.reset()
    
    action = np.zeros(8)
    _, reward, _, _, _ = wrapped_env.step(action)
    assert isinstance(reward, float)


def test_aerial_training_wrapper():
    """Test aerial training wrapper."""
    env = RocketSimEnv()
    wrapped_env = AerialTrainingWrapper(
        env,
        aerial_spawn_probability=1.0,  # Always spawn aerial
        min_ball_height=300.0,
        max_ball_height=1500.0
    )
    
    obs = wrapped_env.reset()
    assert obs is not None
    
    # Step should work normally
    action = np.zeros(8)
    obs, _, _, _, _ = wrapped_env.step(action)
    assert obs is not None


def test_boost_management_wrapper():
    """Test boost management wrapper."""
    env = RocketSimEnv()
    wrapped_env = BoostManagementWrapper(
        env,
        boost_efficiency_weight=0.1,
        boost_starve_penalty=-0.2,
        boost_starve_threshold=20.0
    )
    
    obs = wrapped_env.reset()
    assert obs is not None
    
    action = np.zeros(8)
    obs, reward, _, _, _ = wrapped_env.step(action)
    
    # Reward should include boost management component
    assert isinstance(reward, float)


def test_multiple_wrappers():
    """Test stacking multiple wrappers."""
    env = RocketSimEnv()
    
    # Stack multiple wrappers
    env = NormalizeObservation(env)
    env = FrameStack(env, num_stack=2)
    env = RewardShaping(env, reward_scale=1.5)
    
    obs = env.reset()
    assert obs is not None
    
    # Should be stacked size
    base_size = RocketSimEnv().encoder.feature_size
    assert obs.shape[0] == base_size * 2
    
    action = np.zeros(8)
    obs, reward, _, _, _ = env.step(action)
    assert obs.shape[0] == base_size * 2


def test_wrapper_attribute_delegation():
    """Test that wrappers properly delegate attributes."""
    env = RocketSimEnv()
    wrapped_env = NormalizeObservation(env)
    
    # Should be able to access base env attributes
    assert wrapped_env.game_mode == env.game_mode
    assert wrapped_env.tick_skip == env.tick_skip
    assert wrapped_env.encoder is not None


def test_frame_stack_history_management():
    """Test frame stack history management."""
    env = RocketSimEnv()
    wrapped_env = FrameStack(env, num_stack=3)
    
    wrapped_env.reset()
    
    # Verify frames deque
    assert len(wrapped_env.frames) == 3
    
    # Step multiple times
    action = np.zeros(8)
    for _ in range(5):
        wrapped_env.step(action)
    
    # Should maintain max length
    assert len(wrapped_env.frames) == 3
