"""Recurrent PPO implementation with LSTM/GRU support.

This module extends the standard PPO algorithm to handle recurrent policies
with proper hidden state management and sequence-based training.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from core.models.ppo import PPO


class RecurrentPPO(PPO):
    """Recurrent PPO for LSTM/GRU policies.
    
    Extends PPO with:
    - Hidden state management
    - Sequence-based training (TBPTT)
    - Episode boundary handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        use_amp: bool = False,
    ):
        """Initialize Recurrent PPO.
        
        Args:
            model: Recurrent actor-critic network
            config: Training configuration
            use_amp: Use automatic mixed precision
        """
        super().__init__(model, config, use_amp)
        
        # Recurrent-specific hyperparameters
        self.sequence_length = self.config.get("sequence_length", 16)
        self.truncate_bptt = self.config.get("truncate_bptt", True)
        
        # Hidden state storage
        self.hidden_states = {}
        
    def reset_hidden_states(self, env_ids: Optional[List[int]] = None):
        """Reset LSTM hidden states for specified environments.
        
        Args:
            env_ids: List of environment IDs to reset (None = reset all)
        """
        if env_ids is None:
            self.hidden_states = {}
        else:
            for env_id in env_ids:
                if env_id in self.hidden_states:
                    del self.hidden_states[env_id]
    
    def get_hidden_state(self, env_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get hidden state for an environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Hidden state tuple (h, c) or None
        """
        return self.hidden_states.get(env_id, None)
    
    def set_hidden_state(
        self,
        env_id: int,
        hidden: Tuple[torch.Tensor, ...]
    ):
        """Set hidden state for an environment.
        
        Args:
            env_id: Environment ID
            hidden: Hidden state tuple (h, c) for LSTM or (h,) for GRU
        """
        # Detach to prevent backprop through time beyond episode boundaries
        if isinstance(hidden, tuple):
            self.hidden_states[env_id] = tuple(h.detach() for h in hidden)
        else:
            self.hidden_states[env_id] = hidden.detach()
    
    def update_recurrent(
        self,
        sequences: List[Dict[str, torch.Tensor]],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform recurrent PPO update with sequence-based training.
        
        Args:
            sequences: List of sequence dictionaries containing:
                - observations: (seq_len, obs_size)
                - actions_cat: (seq_len, n_cat)
                - actions_ber: (seq_len, n_ber)
                - old_log_probs_cat: (seq_len, n_cat)
                - old_log_probs_ber: (seq_len, n_ber)
                - old_values: (seq_len,)
                - hidden_init: Initial hidden state
                - dones: (seq_len,) - Episode boundaries
            advantages: Computed advantages for all sequences
            returns: Computed returns for all sequences
            
        Returns:
            Dictionary with loss statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        epoch_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "clip_fraction": [],
        }
        
        # Multiple epochs over the sequences
        for epoch in range(self.n_epochs):
            # Shuffle sequences
            indices = np.random.permutation(len(sequences))
            
            for idx in indices:
                seq = sequences[idx]
                
                # Extract sequence data
                obs = seq["observations"]
                actions_cat = seq["actions_cat"]
                actions_ber = seq["actions_ber"]
                old_log_probs_cat = seq["old_log_probs_cat"]
                old_log_probs_ber = seq["old_log_probs_ber"]
                old_values = seq["old_values"]
                hidden_init = seq.get("hidden_init", None)
                dones = seq.get("dones", None)
                
                # Get advantages and returns for this sequence
                seq_len = obs.shape[0]
                seq_start_idx = idx * self.sequence_length
                seq_end_idx = seq_start_idx + seq_len
                seq_advantages = advantages[seq_start_idx:seq_end_idx]
                seq_returns = returns[seq_start_idx:seq_end_idx]
                
                # Forward pass through sequence
                hidden = hidden_init
                seq_policy_loss = 0.0
                seq_value_loss = 0.0
                seq_entropy_loss = 0.0
                
                for t in range(seq_len):
                    obs_t = obs[t:t+1]
                    
                    # Reset hidden state at episode boundaries if specified
                    if dones is not None and t > 0 and dones[t-1]:
                        hidden = None
                    
                    # Forward pass with optional mixed precision
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            cat_probs, ber_probs, value, _, hidden = self.model(obs_t, hidden)
                    else:
                        cat_probs, ber_probs, value, _, hidden = self.model(obs_t, hidden)
                    
                    # Compute step losses
                    step_policy_loss, step_value_loss, step_entropy_loss, _ = self._compute_ppo_losses(
                        cat_probs,
                        ber_probs,
                        value,
                        actions_cat[t:t+1],
                        actions_ber[t:t+1],
                        old_log_probs_cat[t:t+1],
                        old_log_probs_ber[t:t+1],
                        seq_advantages[t:t+1],
                        seq_returns[t:t+1],
                        old_values[t:t+1],
                    )
                    
                    seq_policy_loss += step_policy_loss
                    seq_value_loss += step_value_loss
                    seq_entropy_loss += step_entropy_loss
                    
                    # Truncate BPTT if enabled
                    if self.truncate_bptt and (t + 1) % self.sequence_length == 0:
                        hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Average losses over sequence
                seq_policy_loss /= seq_len
                seq_value_loss /= seq_len
                seq_entropy_loss /= seq_len
                
                # Total loss
                total_loss = (
                    seq_policy_loss + 
                    self.vf_coef * seq_value_loss + 
                    self.ent_coef * seq_entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track stats
                epoch_stats["policy_loss"].append(seq_policy_loss.item())
                epoch_stats["value_loss"].append(seq_value_loss.item())
                epoch_stats["entropy_loss"].append(seq_entropy_loss.item())
                epoch_stats["total_loss"].append(total_loss.item())
        
        # Anneal entropy coefficient
        self.anneal_entropy()
        self.update_count += 1
        
        # Average stats across epochs
        stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        
        # Store in training stats
        for k, v in stats.items():
            self.training_stats[k].append(v)
        
        return stats
