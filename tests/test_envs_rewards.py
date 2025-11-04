import pytest
import numpy as np
from unittest.mock import Mock

from training.envs.env import RLBotEnv
from training.rewards.rewards import (
    CombinedReward, TouchQualityReward,
    ShotQualityReward, RecoveryReward, StyleReward
)
from rlgym.utils.gamestates import GameState

@pytest.fixture
def game_state():
    state = Mock(spec=GameState)
    state.last_touch = 0
    state.game_time = 0.0
    return state

@pytest.fixture
def previous_action():
    return np.zeros(8)  # Standard RL control action space

def test_rlbot_env_creation():
    """Test environment creation with different scenarios"""
    scenarios = ['aerial', 'wall_carry', 'dribble', 'ceiling']
    
    for scenario in scenarios:
        env = RLBotEnv(
            scenario=scenario,
            reward_fn=Mock(),
            obs_builder=Mock(),
            state_setter=Mock(),
            terminal_condition=Mock()
        )
        assert env.scenario == scenario

def test_touch_quality_reward(game_state, previous_action):
    reward_fn = TouchQualityReward()
    
    # Test no touch
    game_state.last_touch = 1  # Different player
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward == 0.0
    
    # Test basic touch
    game_state.last_touch = 0  # Our player
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward > 0.0

def test_shot_quality_reward(game_state, previous_action):
    reward_fn = ShotQualityReward()
    
    # Test non-shot touch
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward == 0.0
    
    # Mock shot on goal
    game_state._mock_shot_on_goal = True
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward > 0.0

def test_recovery_reward(game_state, previous_action):
    reward_fn = RecoveryReward()
    
    # Test normal state
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward >= 0.0
    
    # Test recovery state
    game_state.game_time = 1.0
    reward_fn.recovery_start = 0.0
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward < 0.0

def test_style_reward(game_state, previous_action):
    reward_fn = StyleReward()
    
    # Test basic play
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward == 0.0
    
    # Test aerial play
    game_state._mock_aerial = True
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward > 0.0

def test_combined_reward(game_state, previous_action):
    weights = {
        'touch': 1.0,
        'shot': 2.0,
        'recovery': 0.5,
        'style': 0.3
    }
    reward_fn = CombinedReward(weights)
    
    # Test basic reward calculation
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert isinstance(reward, float)
    
    # Test weight scaling
    game_state.last_touch = 0
    game_state._mock_shot_on_goal = True
    reward = reward_fn.get_reward(0, game_state, previous_action)
    assert reward > 0.0