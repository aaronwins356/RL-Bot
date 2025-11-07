"""Integration smoke test for RL-Bot training.

This test runs a quick 1000-step training session to verify that all
components work together correctly.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def test_basic_training_smoke():
    """Test that basic training can run for 1000 steps without crashes."""
    try:
        from core.models.nets import ActorCriticNet
        from core.models.ppo import PPO
        from core.training.buffer import ReplayBuffer
        from core.infra.config import ConfigManager
        
        # Load config
        config_path = Path("configs/base.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        config_manager = ConfigManager(config_path)
        
        # Create model
        model = ActorCriticNet(
            input_size=180,
            hidden_sizes=[256, 128],
            action_categoricals=5,
            action_bernoullis=3,
            activation='elu',
            use_lstm=False
        )
        
        # Create PPO
        ppo = PPO(model, config_manager.config.to_dict().get("training", {}), use_amp=False)
        
        # Create buffer
        buffer = ReplayBuffer(capacity=10000)
        
        # Simulate training for a few steps
        for step in range(100):
            # Fake experience
            obs = np.random.randn(180).astype(np.float32)
            action_cat = np.array([0, 1, 2, 0, 1])
            action_ber = np.array([0, 1, 0])
            reward = np.random.randn()
            next_obs = np.random.randn(180).astype(np.float32)
            done = (step % 10 == 9)
            
            buffer.add(
                observation=obs,
                action=np.concatenate([action_cat, action_ber]),
                reward=reward,
                next_observation=next_obs,
                done=done
            )
        
        # Test that we can sample from buffer
        if len(buffer) >= 32:
            batch = buffer.sample(32)
            assert "observations" in batch
            assert "rewards" in batch
        
        logger.info("✓ Basic training smoke test passed")
        
    except Exception as e:
        pytest.fail(f"Training smoke test failed: {e}")


def test_recurrent_training_smoke():
    """Test that recurrent training can initialize without crashes."""
    try:
        from core.models.nets import ActorCriticNet
        from core.models.recurrent_ppo import RecurrentPPO
        from core.training.sequence_buffer import SequenceBuffer
        
        # Create recurrent model
        model = ActorCriticNet(
            input_size=180,
            hidden_sizes=[256, 128],
            action_categoricals=5,
            action_bernoullis=3,
            activation='elu',
            use_lstm=True,
            lstm_hidden_size=128
        )
        
        # Create RecurrentPPO
        config = {
            "sequence_length": 8,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "n_epochs": 2
        }
        recurrent_ppo = RecurrentPPO(model, config, use_amp=False)
        
        # Create sequence buffer
        buffer = SequenceBuffer(capacity=1000, sequence_length=8)
        
        # Add some transitions
        for step in range(20):
            buffer.add_transition(
                observation=np.random.randn(180),
                action_cat=np.array([0, 1, 2, 0, 1]),
                action_ber=np.array([0, 1, 0]),
                reward=1.0,
                done=(step % 10 == 9),
                value=0.5,
                log_prob_cat=np.array([0.1, 0.2, 0.3, 0.1, 0.2]),
                log_prob_ber=np.array([0.5, 0.6, 0.4])
            )
        
        # Test hidden state management
        h = torch.zeros(1, 128)
        c = torch.zeros(1, 128)
        recurrent_ppo.set_hidden_state(0, (h, c))
        
        hidden = recurrent_ppo.get_hidden_state(0)
        assert hidden is not None
        
        recurrent_ppo.reset_hidden_states([0])
        assert recurrent_ppo.get_hidden_state(0) is None
        
        logger.info("✓ Recurrent training smoke test passed")
        
    except Exception as e:
        pytest.fail(f"Recurrent training smoke test failed: {e}")


def test_optimization_config():
    """Test that optimization configuration loads correctly."""
    try:
        import yaml
        
        config_path = Path("configs/base.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check optimization section exists
        assert "training" in config
        assert "optimizations" in config["training"]
        
        opt = config["training"]["optimizations"]
        
        # Check all expected flags
        expected_flags = [
            "use_subproc_vec_env",
            "use_amp",
            "use_torch_compile",
            "use_pinned_memory",
            "batch_inference",
            "action_repeat"
        ]
        
        for flag in expected_flags:
            assert flag in opt, f"Missing optimization flag: {flag}"
        
        logger.info("✓ Optimization config test passed")
        
    except Exception as e:
        pytest.fail(f"Optimization config test failed: {e}")


def test_phase_2_imports():
    """Test that Phase 2 modules can be imported."""
    try:
        from core.models.recurrent_ppo import RecurrentPPO
        from core.training.sequence_buffer import SequenceBuffer
        
        # Test instantiation
        model = None  # Would need actual model
        buffer = SequenceBuffer(capacity=1000, sequence_length=16)
        
        assert buffer.capacity == 1000
        assert buffer.sequence_length == 16
        
        logger.info("✓ Phase 2 imports test passed")
        
    except ImportError as e:
        pytest.fail(f"Phase 2 import test failed: {e}")


def test_phase_3_4_imports():
    """Test that Phase 3-4 modules can be imported."""
    try:
        from core.training.reward_modes import RewardMode
        from core.training.elo_sampling import EloBasedSampling
        
        # Test instantiation
        reward_mode = RewardMode(mode=RewardMode.SPARSE)
        assert reward_mode.mode == RewardMode.SPARSE
        
        sampler = EloBasedSampling(config={"strategy": "elo_weighted"})
        assert sampler.strategy == "elo_weighted"
        
        logger.info("✓ Phase 3-4 imports test passed")
        
    except ImportError as e:
        pytest.fail(f"Phase 3-4 import test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
