"""
Modular RL agent that combines learned policy with hardcoded behaviors.
This agent can use both neural network decisions and heuristic overrides.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from rlgym_rocket_league.utils.gamestates import GameState, PlayerData

from rl_bot.core.model import ActorCritic
from rl_bot.core.behaviors import BehaviorCoordinator


class ModularAgent:
    """
    Agent that combines learned policy with modular behaviors.
    Behaviors can override policy decisions when appropriate.
    """
    
    def __init__(
        self,
        model: ActorCritic,
        action_parser,
        device: torch.device,
        config: Dict[str, Any] = None,
        use_behaviors: bool = True
    ):
        """
        Args:
            model: Neural network policy (ActorCritic)
            action_parser: Action parser from environment
            device: PyTorch device
            config: Configuration dictionary
            use_behaviors: Whether to use behavior overrides
        """
        self.model = model
        self.action_parser = action_parser
        self.device = device
        self.config = config or {}
        self.use_behaviors = use_behaviors
        
        # Initialize behavior coordinator
        if self.use_behaviors:
            self.behavior_coordinator = BehaviorCoordinator(config)
        else:
            self.behavior_coordinator = None
        
        # Statistics
        self.behavior_override_count = 0
        self.total_action_count = 0
    
    def predict(
        self,
        obs: np.ndarray,
        player: Optional[PlayerData] = None,
        state: Optional[GameState] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action, potentially using behavior override.
        
        Args:
            obs: Observation array
            player: Player data (for behavior checks)
            state: Game state (for behavior checks)
            deterministic: Use deterministic policy
            
        Returns:
            action: Action index
            info: Dictionary with metadata
        """
        self.total_action_count += 1
        info = {'behavior_override': False, 'behavior_type': None}
        
        # Check if any behavior wants to override
        if self.use_behaviors and player is not None and state is not None:
            behavior_action = self.behavior_coordinator.get_behavior_action(
                player, state, self.action_parser
            )
            
            if behavior_action is not None and behavior_action.action_index is not None:
                self.behavior_override_count += 1
                info['behavior_override'] = True
                info['behavior_type'] = behavior_action.metadata.get('behavior', 'unknown')
                info['confidence'] = behavior_action.confidence
                
                action = np.array([behavior_action.action_index])
                return action, info
        
        # Use learned policy
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(obs_tensor, deterministic)
        
        action_np = action.cpu().numpy()
        if len(action_np.shape) > 0:
            action_np = action_np[0] if action_np.shape[0] == 1 else action_np
        
        info['value'] = value.cpu().numpy()[0] if hasattr(value, 'cpu') else value
        
        return np.array([action_np]), info
    
    def get_override_rate(self) -> float:
        """Get percentage of actions that were behavior overrides."""
        if self.total_action_count == 0:
            return 0.0
        return self.behavior_override_count / self.total_action_count
    
    def reset(self):
        """Reset agent state."""
        if self.behavior_coordinator:
            self.behavior_coordinator.reset()
        self.behavior_override_count = 0
        self.total_action_count = 0


class HeuristicBaselineAgent:
    """
    Baseline agent using only heuristics (no learned policy).
    Useful for bootstrapping and baseline comparisons.
    """
    
    def __init__(
        self,
        action_parser,
        config: Dict[str, Any] = None
    ):
        """
        Args:
            action_parser: Action parser from environment
            config: Configuration dictionary
        """
        self.action_parser = action_parser
        self.config = config or {}
        
        # Always use behaviors for heuristic agent
        self.behavior_coordinator = BehaviorCoordinator(config)
        
        # Default action for when no behavior activates
        self.default_action_idx = 0  # No input
    
    def predict(
        self,
        obs: np.ndarray,
        player: Optional[PlayerData] = None,
        state: Optional[GameState] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action using only heuristics.
        
        Args:
            obs: Observation array
            player: Player data
            state: Game state
            deterministic: Ignored (heuristics are deterministic)
            
        Returns:
            action: Action index
            info: Dictionary with metadata
        """
        info = {'behavior_override': False, 'behavior_type': 'default'}
        
        # Try to get action from behaviors
        if player is not None and state is not None:
            behavior_action = self.behavior_coordinator.get_behavior_action(
                player, state, self.action_parser
            )
            
            if behavior_action is not None and behavior_action.action_index is not None:
                info['behavior_override'] = True
                info['behavior_type'] = behavior_action.metadata.get('behavior', 'unknown')
                info['confidence'] = behavior_action.confidence
                
                return np.array([behavior_action.action_index]), info
        
        # Default: simple ball chase
        action_idx = self._simple_ball_chase(obs)
        info['behavior_type'] = 'ball_chase'
        
        return np.array([action_idx]), info
    
    def _simple_ball_chase(self, obs: np.ndarray) -> int:
        """
        Simple ball chase heuristic based on observation.
        Obs includes ball relative position.
        """
        # Observation structure (from SimpleObsBuilder):
        # [0:3] player pos, [3:6] player vel, [6:15] rotation, [15:18] ang_vel
        # [18] boost, [19] on_ground, [20:23] ball pos, [23:26] ball vel, [26:29] ball ang_vel
        # [29:32] ball_rel_pos, [32:35] ball_rel_vel
        
        if len(obs) < 32:
            return self.default_action_idx
        
        # Get relative ball position
        ball_rel_x = obs[29]  # Left/right
        ball_rel_y = obs[30]  # Forward/back
        
        # Simple steering toward ball
        # Throttle forward, steer toward ball
        # Action: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        
        # Find action closest to: [1, steer_direction, 0, 0, 0, 0, 1, 0]
        steer_direction = np.clip(ball_rel_x * 3, -1, 1)
        
        desired_action = [1, steer_direction, 0, 0, 0, 0, 1, 0]
        action_idx = self._find_closest_action(desired_action)
        
        return action_idx
    
    def _find_closest_action(self, desired_action):
        """Find closest discrete action."""
        if not hasattr(self.action_parser, 'actions'):
            return 0
        
        actions = self.action_parser.actions
        min_dist = float('inf')
        best_idx = 0
        
        for idx, action in enumerate(actions):
            dist = sum((desired_action[i] - action[i])**2 for i in range(min(3, len(action))))
            if len(action) > 6 and desired_action[6] == action[6]:
                dist -= 0.5
            
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        return best_idx
    
    def reset(self):
        """Reset agent state."""
        self.behavior_coordinator.reset()
