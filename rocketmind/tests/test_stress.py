"""
Comprehensive stress tests and validation for RocketMind.
Tests environment, training, recovery, and RLBot integration.
"""

import unittest
import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEnvironmentStress(unittest.TestCase):
    """Stress test the environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_episodes = 10
        self.max_steps = 1000
    
    def test_environment_stability(self):
        """Test that environment can run multiple episodes without crashing."""
        try:
            from rocketmind.envs import create_rocket_env
            
            config = {
                'environment': {
                    'team_size': 1,
                    'tick_skip': 8,
                    'spawn_opponents': True,
                    'timeout_seconds': 300
                }
            }
            
            env = create_rocket_env(config)
            
            for episode in range(self.num_episodes):
                obs, info = env.reset()
                done = False
                steps = 0
                
                while not done and steps < self.max_steps:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
            
            env.close()
            self.assertTrue(True, "Environment completed stress test")
            
        except ImportError as e:
            self.skipTest(f"Environment not available: {e}")
    
    def test_vectorized_environment(self):
        """Test vectorized environment."""
        try:
            from rocketmind.envs import create_vec_env
            
            config = {
                'environment': {
                    'team_size': 1,
                    'tick_skip': 8,
                    'spawn_opponents': True,
                    'num_envs': 2
                }
            }
            
            env = create_vec_env(config, num_envs=2)
            
            # Run a few steps
            obs, info = env.reset()
            for _ in range(100):
                actions = np.array([env.action_space.sample() for _ in range(2)])
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            env.close()
            self.assertTrue(True, "Vectorized environment works")
            
        except ImportError as e:
            self.skipTest(f"Environment not available: {e}")


class TestTrainingRecovery(unittest.TestCase):
    """Test training recovery and checkpoint management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
    
    def test_checkpoint_save_and_load(self):
        """Test checkpoint saving and loading."""
        from rocketmind.checkpoint_manager import CheckpointManager
        from rocketmind.ppo_core import create_actor_critic
        
        # Create manager
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            keep_top_k=3,
            auto_cleanup=False
        )
        
        # Create dummy model
        config = {
            'network': {
                'hidden_sizes': [64, 64],
                'activation': 'relu',
                'use_layer_norm': False,
                'orthogonal_init': True,
                'use_lstm': False,
                'use_torch_compile': False
            }
        }
        
        model = create_actor_critic(107, 90, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            timestep=1000,
            metrics={'mean_reward': 5.0, 'win_rate': 0.6},
            model_config=config,
            training_config={'batch_size': 4096},
            force=True
        )
        
        self.assertIsNotNone(checkpoint_path)
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['timestep'], 1000)
        self.assertIn('model_state_dict', loaded)
    
    def test_top_k_retention(self):
        """Test that only top-K checkpoints are kept."""
        from rocketmind.checkpoint_manager import CheckpointManager
        from rocketmind.ppo_core import create_actor_critic
        
        # Create manager with keep_top_k=3
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            keep_top_k=3,
            auto_cleanup=True
        )
        
        config = {
            'network': {
                'hidden_sizes': [64, 64],
                'activation': 'relu',
                'use_layer_norm': False,
                'orthogonal_init': True,
                'use_lstm': False,
                'use_torch_compile': False
            }
        }
        
        model = create_actor_critic(107, 90, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # Save 5 checkpoints with different rewards
        rewards = [1.0, 5.0, 3.0, 8.0, 2.0]
        for i, reward in enumerate(rewards):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                timestep=i * 1000,
                metrics={'mean_reward': reward, 'win_rate': 0.5},
                model_config=config,
                training_config={'batch_size': 4096},
                force=True
            )
        
        # Should only have 4 checkpoints (top-3 + most recent)
        # But we just saved 5, so cleanup should have happened
        remaining = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        self.assertLessEqual(len(remaining), 4)
    
    def test_rollback_to_last_good(self):
        """Test rollback functionality."""
        from rocketmind.checkpoint_manager import CheckpointManager
        from rocketmind.ppo_core import create_actor_critic
        
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            keep_top_k=3
        )
        
        config = {
            'network': {
                'hidden_sizes': [64, 64],
                'activation': 'relu',
                'use_layer_norm': False,
                'orthogonal_init': True,
                'use_lstm': False,
                'use_torch_compile': False
            }
        }
        
        model = create_actor_critic(107, 90, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # Save a checkpoint
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            timestep=1000,
            metrics={'mean_reward': 5.0, 'win_rate': 0.6},
            model_config=config,
            training_config={'batch_size': 4096},
            force=True
        )
        
        # Test rollback
        rollback_checkpoint = manager.rollback_to_last_good()
        self.assertIsNotNone(rollback_checkpoint)


class TestAgentInterface(unittest.TestCase):
    """Test agent interface."""
    
    def test_agent_creation(self):
        """Test creating PPO agent."""
        from rocketmind.ppo_core import create_actor_critic, PPOAgent
        
        config = {
            'network': {
                'hidden_sizes': [64, 64],
                'activation': 'relu',
                'use_layer_norm': False,
                'orthogonal_init': True,
                'use_lstm': False,
                'use_torch_compile': False
            }
        }
        
        model = create_actor_critic(107, 90, config)
        agent = PPOAgent(model, deterministic=False)
        
        self.assertIsNotNone(agent)
    
    def test_agent_action_selection(self):
        """Test agent action selection."""
        from rocketmind.ppo_core import create_actor_critic, PPOAgent
        
        config = {
            'network': {
                'hidden_sizes': [64, 64],
                'activation': 'relu',
                'use_layer_norm': False,
                'orthogonal_init': True,
                'use_lstm': False,
                'use_torch_compile': False
            }
        }
        
        model = create_actor_critic(107, 90, config)
        agent = PPOAgent(model, deterministic=True)
        
        # Test single observation
        obs = np.random.randn(107)
        action = agent.select_action(obs)
        
        self.assertEqual(action.shape, ())
        
        # Test batched observation
        obs_batch = np.random.randn(4, 107)
        actions = agent.select_action(obs_batch)
        
        self.assertEqual(actions.shape, (4,))


class TestHyperparameterAdaptation(unittest.TestCase):
    """Test adaptive hyperparameters."""
    
    def test_adaptive_hyperparameters(self):
        """Test that hyperparameters adapt to training."""
        from rocketmind.ppo_core.hyperopt import AdaptiveHyperparameters
        
        config = {
            'training': {
                'learning_rate': 3e-4,
                'ent_coef': 0.01,
                'clip_range': 0.2,
                'target_kl': 0.02,
                'lr_min': 1e-6,
                'lr_max': 1e-3,
                'entropy_min': 0.001,
                'entropy_max': 0.1
            }
        }
        
        adaptive = AdaptiveHyperparameters(config)
        
        # Simulate improving performance
        for i in range(50):
            metrics = {
                'mean_reward': 5.0 + i * 0.1,
                'approx_kl': 0.015,
                'policy_loss': 0.1
            }
            updated = adaptive.update(metrics)
        
        # Check that entropy decreased (exploration reduced)
        self.assertLess(adaptive.entropy_coef, config['training']['ent_coef'])
    
    def test_curriculum_manager(self):
        """Test curriculum learning stages."""
        from rocketmind.ppo_core.hyperopt import CurriculumManager
        
        config = {
            'curriculum': {
                'enabled': True,
                'stages': [
                    {'name': 'basic', 'timesteps': 0, 'opponent_skill': 0.3},
                    {'name': 'intermediate', 'timesteps': 1000, 'opponent_skill': 0.6},
                    {'name': 'advanced', 'timesteps': 2000, 'opponent_skill': 1.0}
                ]
            }
        }
        
        curriculum = CurriculumManager(config)
        
        # Should start at basic stage
        self.assertEqual(curriculum.get_stage_name(), 'basic')
        
        # Advance to intermediate
        curriculum.update(1500)
        self.assertEqual(curriculum.get_stage_name(), 'intermediate')
        
        # Advance to advanced
        curriculum.update(2500)
        self.assertEqual(curriculum.get_stage_name(), 'advanced')


def run_stress_tests():
    """Run all stress tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentStress))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestHyperparameterAdaptation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_stress_tests()
    sys.exit(0 if success else 1)
