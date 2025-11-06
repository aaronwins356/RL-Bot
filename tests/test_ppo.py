"""Unit tests for PPO algorithm."""
import torch
import numpy as np
import pytest

from core.models.nets import ActorCriticNet
from core.models.ppo import PPO


def test_ppo_initialization():
    """Test PPO initialization."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    ppo = PPO(model)
    assert ppo is not None
    assert ppo.learning_rate > 0
    assert ppo.clip_range > 0


def test_ppo_compute_gae():
    """Test GAE computation."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    ppo = PPO(model)
    
    # Create dummy trajectory
    rewards = np.array([1.0, 0.5, 0.0, -0.5, 1.0])
    values = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
    dones = np.array([False, False, False, False, True])
    next_value = 0.0
    
    advantages, returns = ppo.compute_gae(rewards, values, dones, next_value)
    
    # Check shapes
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    
    # Check values are reasonable
    assert not np.any(np.isnan(advantages))
    assert not np.any(np.isnan(returns))


def test_ppo_update_shapes():
    """Test PPO update with correct shapes."""
    model = ActorCriticNet(
        input_size=173,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3
    )
    
    ppo = PPO(model, {"n_epochs": 1})  # Just 1 epoch for testing
    
    batch_size = 32
    obs_size = 173
    
    # Create dummy data
    observations = torch.randn(batch_size, obs_size)
    actions_cat = torch.randint(0, 3, (batch_size, 5))
    actions_ber = torch.randint(0, 2, (batch_size, 3))
    
    # Get log probs from model
    with torch.no_grad():
        cat_probs, ber_probs, values, _, _ = model(observations)
        
        # Compute log probs
        old_log_probs_cat = torch.log(
            torch.gather(cat_probs, 2, actions_cat.unsqueeze(2)) + 1e-8
        ).squeeze(2)
        old_log_probs_ber = torch.log(
            torch.gather(ber_probs, 2, actions_ber.unsqueeze(2)) + 1e-8
        ).squeeze(2)
        old_values = values.squeeze()
    
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    
    # Update (should not crash)
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
    
    # Check stats
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "entropy_loss" in stats
    assert isinstance(stats["policy_loss"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
