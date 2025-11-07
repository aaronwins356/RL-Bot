"""Unit tests for recurrent PPO and sequence buffer."""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.recurrent_ppo import RecurrentPPO
from core.training.sequence_buffer import SequenceBuffer
from core.models.nets import ActorCriticNet


class TestSequenceBuffer:
    """Tests for SequenceBuffer."""
    
    def test_buffer_initialization(self):
        """Test buffer can be initialized."""
        buffer = SequenceBuffer(capacity=10000, sequence_length=16)
        assert buffer.capacity == 10000
        assert buffer.sequence_length == 16
        assert len(buffer) == 0
    
    def test_add_transition(self):
        """Test adding transitions to buffer."""
        buffer = SequenceBuffer(capacity=1000, sequence_length=4)
        
        # Add some transitions
        for i in range(5):
            buffer.add_transition(
                observation=np.random.randn(180),
                action_cat=np.array([0, 1, 2, 0, 1]),
                action_ber=np.array([0, 1, 0]),
                reward=1.0,
                done=(i == 4),
                value=0.5,
                log_prob_cat=np.array([0.1, 0.2, 0.3, 0.1, 0.2]),
                log_prob_ber=np.array([0.5, 0.6, 0.4])
            )
        
        assert len(buffer) == 5
        assert buffer.episodes_stored == 1
    
    def test_sequence_creation(self):
        """Test sequence dictionary creation."""
        buffer = SequenceBuffer(capacity=1000, sequence_length=4, store_full_episodes=False)
        
        # Add transitions to form one sequence
        for i in range(4):
            buffer.add_transition(
                observation=np.random.randn(180),
                action_cat=np.array([0, 1, 2, 0, 1]),
                action_ber=np.array([0, 1, 0]),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob_cat=np.array([0.1, 0.2, 0.3, 0.1, 0.2]),
                log_prob_ber=np.array([0.5, 0.6, 0.4])
            )
        
        sequences = buffer.get_sequences()
        assert len(sequences) == 1
        
        seq = sequences[0]
        assert seq["observations"].shape == (4, 180)
        assert seq["actions_cat"].shape == (4, 5)
        assert seq["actions_ber"].shape == (4, 3)
        assert seq["rewards"].shape == (4,)
        assert seq["dones"].shape == (4,)
    
    def test_advantage_computation(self):
        """Test advantage computation with GAE."""
        buffer = SequenceBuffer(capacity=1000, sequence_length=4)
        
        # Add an episode
        for i in range(5):
            buffer.add_transition(
                observation=np.random.randn(180),
                action_cat=np.array([0, 1, 2, 0, 1]),
                action_ber=np.array([0, 1, 0]),
                reward=1.0,
                done=(i == 4),
                value=0.5,
                log_prob_cat=np.array([0.1, 0.2, 0.3, 0.1, 0.2]),
                log_prob_ber=np.array([0.5, 0.6, 0.4])
            )
        
        advantages, returns = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)
        assert advantages.shape == (5,)
        assert returns.shape == (5,)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()


class TestRecurrentPPO:
    """Tests for RecurrentPPO."""
    
    def test_recurrent_ppo_initialization(self):
        """Test RecurrentPPO can be initialized."""
        model = ActorCriticNet(
            input_size=180,
            hidden_sizes=[256, 128],
            action_categoricals=5,
            action_bernoullis=3,
            use_lstm=True,
            lstm_hidden_size=128
        )
        
        config = {
            "sequence_length": 16,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "n_epochs": 3
        }
        
        ppo = RecurrentPPO(model, config, use_amp=False)
        assert ppo.sequence_length == 16
        assert ppo.truncate_bptt == True
    
    def test_hidden_state_management(self):
        """Test hidden state reset and retrieval."""
        model = ActorCriticNet(
            input_size=180,
            hidden_sizes=[256, 128],
            action_categoricals=5,
            action_bernoullis=3,
            use_lstm=True,
            lstm_hidden_size=128
        )
        
        ppo = RecurrentPPO(model, {}, use_amp=False)
        
        # Set hidden state
        h = torch.zeros(1, 128)
        c = torch.zeros(1, 128)
        ppo.set_hidden_state(0, (h, c))
        
        # Retrieve hidden state
        hidden = ppo.get_hidden_state(0)
        assert hidden is not None
        assert hidden[0].shape == (1, 128)
        assert hidden[1].shape == (1, 128)
        
        # Reset specific environment
        ppo.reset_hidden_states([0])
        assert ppo.get_hidden_state(0) is None
        
        # Reset all
        ppo.set_hidden_state(1, (h, c))
        ppo.reset_hidden_states()
        assert ppo.get_hidden_state(1) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
