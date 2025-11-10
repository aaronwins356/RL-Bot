"""
Basic tests for RocketMind PPO implementation.
Tests core PPO components, network architecture, and training loop.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rocketmind.ppo_core import (
    ActorCritic,
    create_actor_critic,
    RolloutBuffer,
    ppo_loss,
    value_loss,
    get_device,
    set_seed
)


class TestPPONetwork(unittest.TestCase):
    """Test PPO network architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.obs_dim = 107
        self.action_dim = 90
        self.config = {
            'network': {
                'hidden_sizes': [256, 256],
                'activation': 'relu',
                'use_layer_norm': False,
                'orthogonal_init': True,
                'use_lstm': False,
                'use_torch_compile': False
            }
        }
        set_seed(42)
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = create_actor_critic(self.obs_dim, self.action_dim, self.config)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, ActorCritic)
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        model = create_actor_critic(self.obs_dim, self.action_dim, self.config)
        obs = torch.randn(4, self.obs_dim)
        
        logits, value, _ = model(obs)
        
        self.assertEqual(logits.shape, (4, self.action_dim))
        self.assertEqual(value.shape, (4, 1))
    
    def test_action_sampling(self):
        """Test action sampling."""
        model = create_actor_critic(self.obs_dim, self.action_dim, self.config)
        obs = torch.randn(4, self.obs_dim)
        
        action, log_prob, entropy, value = model.get_action_and_value(obs)
        
        self.assertEqual(action.shape, (4,))
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(value.shape, (4,))
    
    def test_deterministic_action(self):
        """Test deterministic action selection."""
        model = create_actor_critic(self.obs_dim, self.action_dim, self.config)
        obs = torch.randn(1, self.obs_dim)
        
        action1, _, _, _ = model.get_action_and_value(obs, deterministic=True)
        action2, _, _, _ = model.get_action_and_value(obs, deterministic=True)
        
        self.assertEqual(action1.item(), action2.item())


class TestRolloutBuffer(unittest.TestCase):
    """Test rollout buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer_size = 128
        self.obs_dim = 107
        self.num_envs = 4
        self.device = torch.device('cpu')
        
        self.buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            obs_dim=self.obs_dim,
            num_envs=self.num_envs,
            device=self.device,
            gamma=0.99,
            gae_lambda=0.95
        )
    
    def test_buffer_creation(self):
        """Test buffer creation."""
        self.assertEqual(self.buffer.buffer_size, self.buffer_size)
        self.assertEqual(self.buffer.num_envs, self.num_envs)
    
    def test_add_transition(self):
        """Test adding transition to buffer."""
        obs = np.random.randn(self.num_envs, self.obs_dim)
        action = np.random.randint(0, 90, self.num_envs)
        reward = np.random.randn(self.num_envs)
        done = np.zeros(self.num_envs)
        value = np.random.randn(self.num_envs)
        log_prob = np.random.randn(self.num_envs)
        
        self.buffer.add(obs, action, reward, done, value, log_prob)
        
        self.assertEqual(self.buffer.pos, 1)
    
    def test_compute_returns(self):
        """Test GAE computation."""
        # Fill buffer
        for _ in range(self.buffer_size):
            obs = np.random.randn(self.num_envs, self.obs_dim)
            action = np.random.randint(0, 90, self.num_envs)
            reward = np.random.randn(self.num_envs)
            done = np.zeros(self.num_envs)
            value = np.random.randn(self.num_envs)
            log_prob = np.random.randn(self.num_envs)
            
            self.buffer.add(obs, action, reward, done, value, log_prob)
        
        # Compute returns
        last_values = np.random.randn(self.num_envs)
        last_dones = np.zeros(self.num_envs)
        
        self.buffer.compute_returns_and_advantages(last_values, last_dones)
        
        self.assertEqual(self.buffer.returns.shape, (self.buffer_size, self.num_envs))
        self.assertEqual(self.buffer.advantages.shape, (self.buffer_size, self.num_envs))


class TestPPOLosses(unittest.TestCase):
    """Test PPO loss functions."""
    
    def test_ppo_loss(self):
        """Test PPO clipped loss."""
        batch_size = 64
        
        log_probs = torch.randn(batch_size)
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        
        loss, clip_frac, approx_kl = ppo_loss(
            log_probs,
            old_log_probs,
            advantages,
            clip_range=0.2
        )
        
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(clip_frac, 0.0)
        self.assertLessEqual(clip_frac, 1.0)
    
    def test_value_loss(self):
        """Test value function loss."""
        batch_size = 64
        
        values = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        loss = value_loss(values, returns)
        
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss.item(), 0.0)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device('auto')
        self.assertIsNotNone(device)
        
        cpu_device = get_device('cpu')
        self.assertEqual(cpu_device.type, 'cpu')
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        a = torch.randn(5)
        
        set_seed(42)
        b = torch.randn(5)
        
        self.assertTrue(torch.allclose(a, b))


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == '__main__':
    run_tests()
