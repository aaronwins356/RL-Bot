"""Sequence buffer for recurrent policy training.

This module implements a buffer that stores and manages sequences/episodes
for training recurrent policies with LSTM/GRU.
"""
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import deque


class SequenceBuffer:
    """Buffer for storing sequences/episodes for recurrent policy training.
    
    Stores complete episodes or fixed-length sequences with proper
    handling of episode boundaries and LSTM hidden states.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        sequence_length: int = 16,
        store_full_episodes: bool = True
    ):
        """Initialize sequence buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            sequence_length: Length of sequences for training
            store_full_episodes: If True, store full episodes; else fixed sequences
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.store_full_episodes = store_full_episodes
        
        # Storage
        self.episodes = deque(maxlen=capacity // sequence_length)
        self.current_episode = []
        
        # Statistics
        self.episodes_stored = 0
        self.total_transitions = 0
    
    def add_transition(
        self,
        observation: np.ndarray,
        action_cat: np.ndarray,
        action_ber: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob_cat: np.ndarray,
        log_prob_ber: np.ndarray,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        info: Optional[Dict[str, Any]] = None
    ):
        """Add a transition to the current episode.
        
        Args:
            observation: Current observation
            action_cat: Categorical actions taken
            action_ber: Bernoulli actions taken
            reward: Reward received
            done: Whether episode is done
            value: Value estimate
            log_prob_cat: Log probabilities for categorical actions
            log_prob_ber: Log probabilities for bernoulli actions
            hidden_state: LSTM hidden state (h, c)
            info: Additional information
        """
        transition = {
            "observation": observation,
            "action_cat": action_cat,
            "action_ber": action_ber,
            "reward": reward,
            "done": done,
            "value": value,
            "log_prob_cat": log_prob_cat,
            "log_prob_ber": log_prob_ber,
            "hidden_state": hidden_state,
            "info": info or {}
        }
        
        self.current_episode.append(transition)
        self.total_transitions += 1
        
        # If episode is done or we've reached sequence length
        if done or (not self.store_full_episodes and len(self.current_episode) >= self.sequence_length):
            self._store_episode()
    
    def _store_episode(self):
        """Store the current episode in the buffer."""
        if len(self.current_episode) > 0:
            self.episodes.append(list(self.current_episode))
            self.episodes_stored += 1
            self.current_episode = []
    
    def get_sequences(self, num_sequences: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """Get sequences for training.
        
        Args:
            num_sequences: Number of sequences to retrieve (None = all available)
            
        Returns:
            List of sequence dictionaries for training
        """
        sequences = []
        
        if num_sequences is None:
            episodes_to_process = list(self.episodes)
        else:
            # Sample random episodes
            episode_count = min(num_sequences, len(self.episodes))
            indices = np.random.choice(len(self.episodes), episode_count, replace=False)
            episodes_to_process = [self.episodes[i] for i in indices]
        
        for episode in episodes_to_process:
            if self.store_full_episodes:
                # Split episode into sequences of fixed length
                for i in range(0, len(episode), self.sequence_length):
                    seq_data = episode[i:i + self.sequence_length]
                    if len(seq_data) > 0:
                        sequences.append(self._create_sequence_dict(seq_data))
            else:
                # Each episode is already a sequence
                sequences.append(self._create_sequence_dict(episode))
        
        return sequences
    
    def _create_sequence_dict(self, sequence_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Create a sequence dictionary from transition list.
        
        Args:
            sequence_data: List of transitions
            
        Returns:
            Dictionary with batched sequence data
        """
        seq_len = len(sequence_data)
        
        # Stack observations and actions
        observations = torch.stack([
            torch.from_numpy(t["observation"]).float()
            for t in sequence_data
        ])
        
        actions_cat = torch.stack([
            torch.from_numpy(t["action_cat"]).long()
            for t in sequence_data
        ])
        
        actions_ber = torch.stack([
            torch.from_numpy(t["action_ber"]).long()
            for t in sequence_data
        ])
        
        rewards = torch.tensor([t["reward"] for t in sequence_data], dtype=torch.float32)
        dones = torch.tensor([t["done"] for t in sequence_data], dtype=torch.bool)
        values = torch.tensor([t["value"] for t in sequence_data], dtype=torch.float32)
        
        log_probs_cat = torch.stack([
            torch.from_numpy(t["log_prob_cat"]).float()
            for t in sequence_data
        ])
        
        log_probs_ber = torch.stack([
            torch.from_numpy(t["log_prob_ber"]).float()
            for t in sequence_data
        ])
        
        # Get initial hidden state
        hidden_init = sequence_data[0].get("hidden_state", None)
        
        return {
            "observations": observations,
            "actions_cat": actions_cat,
            "actions_ber": actions_ber,
            "rewards": rewards,
            "dones": dones,
            "old_values": values,
            "old_log_probs_cat": log_probs_cat,
            "old_log_probs_ber": log_probs_ber,
            "hidden_init": hidden_init,
            "sequence_length": seq_len
        }
    
    def compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns for all stored sequences using GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns) tensors
        """
        all_advantages = []
        all_returns = []
        
        for episode in self.episodes:
            # Extract rewards, values, and dones
            rewards = np.array([t["reward"] for t in episode])
            values = np.array([t["value"] for t in episode])
            dones = np.array([t["done"] for t in episode])
            
            # Compute GAE
            advantages = np.zeros_like(rewards)
            last_gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0 if dones[t] else values[t]
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                last_gae = delta + gamma * gae_lambda * last_gae * (1 - dones[t])
                advantages[t] = last_gae
            
            returns = advantages + values
            
            all_advantages.extend(advantages.tolist())
            all_returns.extend(returns.tolist())
        
        return (
            torch.tensor(all_advantages, dtype=torch.float32),
            torch.tensor(all_returns, dtype=torch.float32)
        )
    
    def clear(self):
        """Clear the buffer."""
        self.episodes.clear()
        self.current_episode = []
        self.total_transitions = 0
    
    def __len__(self):
        """Return number of transitions stored."""
        return self.total_transitions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer stats
        """
        return {
            "episodes_stored": self.episodes_stored,
            "total_transitions": self.total_transitions,
            "current_episode_length": len(self.current_episode),
            "buffer_episodes": len(self.episodes)
        }
