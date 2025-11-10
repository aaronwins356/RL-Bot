"""
Agent interface for RocketMind PPO Bot.
Provides a clean interface for the PPO agent with action selection and inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .network import ActorCritic
from .utils import get_device, load_checkpoint


class PPOAgent:
    """
    PPO Agent with clean inference interface.
    Supports both deterministic and stochastic action selection.
    """
    
    def __init__(
        self,
        model: ActorCritic,
        device: Optional[torch.device] = None,
        deterministic: bool = False
    ):
        """
        Args:
            model: Actor-Critic network
            device: PyTorch device (auto-detected if None)
            deterministic: Whether to use deterministic actions
        """
        self.model = model
        self.device = device or get_device('auto')
        self.deterministic = deterministic
        self.model.to(self.device)
        self.model.eval()  # Set to eval mode by default
        
    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: Optional[bool] = None
    ) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            observation: Observation from environment (can be batched)
            deterministic: Override default deterministic behavior
            
        Returns:
            Selected action(s)
        """
        # Handle single observation
        single_obs = False
        if observation.ndim == 1:
            observation = observation[np.newaxis, ...]
            single_obs = True
        
        # Convert to tensor
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # Get action
        det = deterministic if deterministic is not None else self.deterministic
        action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=det)
        
        # Convert to numpy
        action = action.cpu().numpy()
        
        # Return single action if input was single
        if single_obs:
            action = action[0]
        
        return action
    
    @torch.no_grad()
    def get_value(self, observation: np.ndarray) -> np.ndarray:
        """
        Get value estimate for observation.
        
        Args:
            observation: Observation from environment (can be batched)
            
        Returns:
            Value estimate(s)
        """
        # Handle single observation
        single_obs = False
        if observation.ndim == 1:
            observation = observation[np.newaxis, ...]
            single_obs = True
        
        # Convert to tensor
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # Get value
        _, value, _ = self.model(obs_tensor)
        
        # Convert to numpy
        value = value.squeeze(-1).cpu().numpy()
        
        # Return single value if input was single
        if single_obs:
            value = value[0]
        
        return value
    
    @torch.no_grad()
    def predict(
        self,
        observation: np.ndarray,
        deterministic: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action with additional info (compatible with stable-baselines3 interface).
        
        Args:
            observation: Observation from environment
            deterministic: Override default deterministic behavior
            
        Returns:
            action: Selected action
            info: Dictionary with additional information (value, log_prob, etc.)
        """
        # Handle single observation
        single_obs = False
        if observation.ndim == 1:
            observation = observation[np.newaxis, ...]
            single_obs = True
        
        # Convert to tensor
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # Get action and info
        det = deterministic if deterministic is not None else self.deterministic
        action, log_prob, entropy, value = self.model.get_action_and_value(obs_tensor, deterministic=det)
        
        # Build info dict
        info = {
            'value': value.cpu().numpy(),
            'log_prob': log_prob.cpu().numpy(),
            'entropy': entropy.cpu().numpy()
        }
        
        # Convert action to numpy
        action = action.cpu().numpy()
        
        # Handle single observation case
        if single_obs:
            action = action[0]
            info = {k: v[0] for k, v in info.items()}
        
        return action, info
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def save(self, path: str):
        """
        Save agent to file.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'deterministic': self.deterministic,
        }
        torch.save(checkpoint, path)
        
    @classmethod
    def load(
        cls,
        path: str,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> 'PPOAgent':
        """
        Load agent from file.
        
        Args:
            path: Path to checkpoint
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: Model configuration
            device: PyTorch device
            
        Returns:
            Loaded agent
        """
        from .network import create_actor_critic
        
        # Create model
        model = create_actor_critic(obs_dim, action_dim, config)
        
        # Load checkpoint
        checkpoint = load_checkpoint(path, model, device=device)
        
        # Create agent
        agent = cls(
            model=model,
            device=device,
            deterministic=checkpoint.get('deterministic', False)
        )
        
        return agent


class MultiAgentWrapper:
    """
    Wrapper for multiple agents (e.g., self-play or team scenarios).
    Manages multiple PPO agents and coordinates their actions.
    """
    
    def __init__(self, agents: Dict[str, PPOAgent]):
        """
        Args:
            agents: Dictionary mapping agent names to PPOAgent instances
        """
        self.agents = agents
        
    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: Optional[bool] = None
    ) -> Dict[str, np.ndarray]:
        """
        Select actions for all agents.
        
        Args:
            observations: Dictionary mapping agent names to observations
            deterministic: Override default deterministic behavior
            
        Returns:
            Dictionary mapping agent names to actions
        """
        actions = {}
        for name, agent in self.agents.items():
            if name in observations:
                actions[name] = agent.select_action(observations[name], deterministic)
        return actions
    
    def train(self):
        """Set all agents to training mode."""
        for agent in self.agents.values():
            agent.train()
    
    def eval(self):
        """Set all agents to evaluation mode."""
        for agent in self.agents.values():
            agent.eval()
