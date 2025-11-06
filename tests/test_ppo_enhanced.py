"""Unit tests for enhanced PPO features."""
import torch
import numpy as np
import pytest

from core.models.nets import ActorCriticNet
from core.models.ppo import PPO


def test_ppo_dynamic_lambda():
    """Test dynamic GAE lambda adjustment."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'use_dynamic_lambda': True,
        'min_gae_lambda': 0.85,
        'max_gae_lambda': 0.98
    }
    
    ppo = PPO(model, config)
    
    rewards = np.array([1.0, 0.5, 0.0, -0.5, 1.0])
    values = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
    dones = np.array([False, False, False, False, True])
    
    # With high explained variance
    advantages_high, returns_high = ppo.compute_gae(
        rewards, values, dones, next_value=0.0, explained_variance=0.9
    )
    
    # With low explained variance
    advantages_low, returns_low = ppo.compute_gae(
        rewards, values, dones, next_value=0.0, explained_variance=0.1
    )
    
    # Both should work without errors
    assert advantages_high.shape == rewards.shape
    assert advantages_low.shape == rewards.shape
    
    # Different explained variances should lead to different advantages
    # (due to different lambda values)
    assert not np.allclose(advantages_high, advantages_low)


def test_ppo_entropy_annealing():
    """Test entropy coefficient annealing."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'use_entropy_annealing': True,
        'ent_coef': 0.01,
        'min_ent_coef': 0.001,
        'ent_anneal_rate': 0.99
    }
    
    ppo = PPO(model, config)
    
    initial_coef = ppo.ent_coef
    
    # Anneal multiple times
    for _ in range(10):
        ppo.anneal_entropy()
    
    # Entropy coefficient should decrease
    assert ppo.ent_coef < initial_coef
    
    # Should not go below minimum
    for _ in range(1000):
        ppo.anneal_entropy()
    
    assert ppo.ent_coef >= config['min_ent_coef']


def test_ppo_reward_scaling():
    """Test reward scaling functionality."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'use_reward_scaling': True
    }
    
    ppo = PPO(model, config)
    
    # Generate some rewards with known statistics
    rewards = np.random.randn(100) * 5 + 10  # mean=10, std=5
    
    # Update scaling statistics multiple times to stabilize
    for _ in range(20):
        batch_rewards = np.random.randn(100) * 5 + 10
        ppo.update_reward_scaling(batch_rewards)
    
    # Scale rewards
    scaled_rewards = ppo.scale_rewards(rewards)
    
    # Scaled rewards should be more standardized than raw rewards
    # (mean should be closer to 0 than original mean of 10)
    assert abs(np.mean(scaled_rewards)) < abs(np.mean(rewards))


def test_ppo_reward_scaling_disabled():
    """Test that reward scaling can be disabled."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'use_reward_scaling': False
    }
    
    ppo = PPO(model, config)
    
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Should return rewards unchanged
    scaled = ppo.scale_rewards(rewards)
    assert np.array_equal(scaled, rewards)


def test_ppo_training_stats_tracking():
    """Test that training stats are properly tracked."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'use_dynamic_lambda': True,
        'use_entropy_annealing': True,
        'use_reward_scaling': True,
        'n_epochs': 1
    }
    
    ppo = PPO(model, config)
    
    # Perform some operations that update stats
    rewards = np.random.randn(50)
    values = np.random.randn(50)
    dones = np.zeros(50, dtype=bool)
    
    ppo.update_reward_scaling(rewards)
    ppo.compute_gae(rewards, values, dones, 0.0, explained_variance=0.5)
    ppo.anneal_entropy()
    
    # Check that stats are being tracked
    assert 'gae_lambda' in ppo.training_stats
    assert 'ent_coef' in ppo.training_stats
    assert 'reward_scale' in ppo.training_stats
    
    assert len(ppo.training_stats['reward_scale']) > 0


def test_ppo_reward_scale_clipping():
    """Test that reward scale is properly clipped."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'use_reward_scaling': True
    }
    
    ppo = PPO(model, config)
    
    # Extreme rewards (very small std)
    rewards_small_std = np.array([1.0, 1.001, 1.002, 0.999, 1.001])
    ppo.update_reward_scaling(rewards_small_std)
    
    # Scale should be clipped to reasonable range
    assert 0.1 <= ppo.reward_scale <= 10.0
    
    # Extreme rewards (very large std)
    rewards_large_std = np.random.randn(100) * 1000
    ppo.update_reward_scaling(rewards_large_std)
    
    # Scale should still be clipped
    assert 0.1 <= ppo.reward_scale <= 10.0


def test_ppo_gae_with_terminal_states():
    """Test GAE computation with terminal states."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    ppo = PPO(model)
    
    # Trajectory with terminal state in the middle
    rewards = np.array([1.0, 0.5, 0.0, -0.5, 1.0])
    values = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
    dones = np.array([False, False, True, False, False])  # Terminal at index 2
    
    advantages, returns = ppo.compute_gae(rewards, values, dones, next_value=0.0)
    
    # Should handle terminal state correctly (bootstrap value = 0)
    assert not np.any(np.isnan(advantages))
    assert not np.any(np.isinf(advantages))


def test_ppo_multiple_updates():
    """Test multiple PPO updates in sequence."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    config = {
        'n_epochs': 1,
        'use_entropy_annealing': True
    }
    
    ppo = PPO(model, config)
    
    batch_size = 32
    
    initial_ent_coef = ppo.ent_coef
    
    for _ in range(3):
        # Create dummy batch
        observations = torch.randn(batch_size, 173)
        actions_cat = torch.randint(0, 3, (batch_size, 5))
        actions_ber = torch.randint(0, 2, (batch_size, 3))
        
        with torch.no_grad():
            cat_probs, ber_probs, values, _, _ = model(observations)
            old_log_probs_cat = torch.log(
                torch.gather(cat_probs, 2, actions_cat.unsqueeze(2)) + 1e-8
            ).squeeze(2)
            old_log_probs_ber = torch.log(
                torch.gather(ber_probs, 2, actions_ber.unsqueeze(2)) + 1e-8
            ).squeeze(2)
            old_values = values.squeeze()
        
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        stats = ppo.update(
            observations,
            actions_cat,
            actions_ber,
            old_log_probs_cat,
            old_log_probs_ber,
            advantages,
            returns,
            old_values
        )
        
        assert stats is not None
    
    # Entropy coefficient should have decreased
    assert ppo.ent_coef < initial_ent_coef


def test_ppo_get_stats():
    """Test getting PPO statistics."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    ppo = PPO(model)
    
    stats = ppo.get_stats()
    
    # Should return valid stats dict
    assert isinstance(stats, dict)
    
    # Check for expected keys (from base implementation)
    # Note: This test may need to be updated based on actual implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
