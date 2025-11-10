"""
Hyperparameter optimization and self-adaptive hyperparameters for PPO.
Inspired by top-tier bots like Nexto, Coconut, and Ripple.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import warnings


class AdaptiveHyperparameters:
    """
    Self-adaptive hyperparameters that adjust based on training progress.
    Implements:
    - Dynamic entropy decay (reduces exploration as performance improves)
    - Adaptive learning rate with performance-based adjustment
    - KL-penalty auto-tuner
    - Adaptive clip range
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration with initial hyperparameters
        """
        self.config = config
        
        # Initial values
        self.learning_rate = config['training']['learning_rate']
        self.entropy_coef = config['training']['ent_coef']
        self.clip_range = config['training']['clip_range']
        self.kl_target = config['training'].get('target_kl', 0.02)
        
        # Adaptation parameters
        self.lr_min = config['training'].get('lr_min', 1e-6)
        self.lr_max = config['training'].get('lr_max', 1e-3)
        self.entropy_min = config['training'].get('entropy_min', 0.001)
        self.entropy_max = config['training'].get('entropy_max', 0.1)
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.kl_history = deque(maxlen=20)
        self.loss_history = deque(maxlen=50)
        
        # Adaptation rates
        self.lr_adapt_rate = 0.1
        self.entropy_adapt_rate = 0.05
        self.kl_adapt_rate = 0.2
        
        # Performance baseline
        self.best_reward = float('-inf')
        self.no_improvement_steps = 0
        
    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update hyperparameters based on training metrics.
        
        Args:
            metrics: Dictionary with training metrics (reward, kl, loss, etc.)
            
        Returns:
            Updated hyperparameters
        """
        # Track metrics
        if 'mean_reward' in metrics:
            self.reward_history.append(metrics['mean_reward'])
        if 'approx_kl' in metrics:
            self.kl_history.append(metrics['approx_kl'])
        if 'policy_loss' in metrics:
            self.loss_history.append(metrics['policy_loss'])
        
        # Adaptive entropy coefficient
        self._update_entropy_coef(metrics)
        
        # Adaptive learning rate
        self._update_learning_rate(metrics)
        
        # Adaptive KL penalty
        self._update_kl_penalty(metrics)
        
        # Adaptive clip range (experimental)
        self._update_clip_range(metrics)
        
        return self.get_current_values()
    
    def _update_entropy_coef(self, metrics: Dict[str, float]):
        """
        Decrease entropy as performance improves (exploration → exploitation).
        """
        if len(self.reward_history) < 10:
            return
        
        # Calculate improvement
        recent_reward = np.mean(list(self.reward_history)[-10:])
        old_reward = np.mean(list(self.reward_history)[:10])
        
        if recent_reward > old_reward:
            # Improving - reduce entropy (less exploration)
            self.entropy_coef *= (1 - self.entropy_adapt_rate)
        else:
            # Not improving - increase entropy (more exploration)
            self.entropy_coef *= (1 + self.entropy_adapt_rate * 0.5)
        
        # Clamp to bounds
        self.entropy_coef = np.clip(self.entropy_coef, self.entropy_min, self.entropy_max)
    
    def _update_learning_rate(self, metrics: Dict[str, float]):
        """
        Adjust learning rate based on performance and stability.
        """
        if len(self.reward_history) < 20:
            return
        
        mean_reward = np.mean(self.reward_history)
        
        # Track best performance
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.no_improvement_steps = 0
        else:
            self.no_improvement_steps += 1
        
        # Increase LR if making good progress
        if self.no_improvement_steps == 0:
            self.learning_rate *= (1 + self.lr_adapt_rate * 0.1)
        # Decrease LR if plateau
        elif self.no_improvement_steps > 10:
            self.learning_rate *= (1 - self.lr_adapt_rate)
        
        # Check loss stability
        if len(self.loss_history) >= 10:
            recent_loss_std = np.std(list(self.loss_history)[-10:])
            if recent_loss_std > 1.0:  # Unstable training
                self.learning_rate *= 0.9  # Reduce LR
        
        # Clamp to bounds
        self.learning_rate = np.clip(self.learning_rate, self.lr_min, self.lr_max)
    
    def _update_kl_penalty(self, metrics: Dict[str, float]):
        """
        Auto-tune KL penalty coefficient to maintain target KL divergence.
        """
        if len(self.kl_history) < 5:
            return
        
        mean_kl = np.mean(self.kl_history)
        
        # Adjust KL coefficient to maintain target
        if 'kl_coef' not in self.config['training']:
            self.config['training']['kl_coef'] = 0.0
        
        kl_coef = self.config['training']['kl_coef']
        
        if mean_kl > self.kl_target * 1.5:
            # KL too high - increase penalty
            kl_coef += self.kl_adapt_rate
        elif mean_kl < self.kl_target * 0.5:
            # KL too low - decrease penalty
            kl_coef = max(0, kl_coef - self.kl_adapt_rate * 0.5)
        
        self.config['training']['kl_coef'] = kl_coef
    
    def _update_clip_range(self, metrics: Dict[str, float]):
        """
        Adjust PPO clip range based on training stability.
        """
        if len(self.kl_history) < 10:
            return
        
        mean_kl = np.mean(self.kl_history)
        
        # If KL is consistently low, can afford larger updates
        if mean_kl < self.kl_target * 0.3:
            self.clip_range = min(0.3, self.clip_range * 1.05)
        # If KL is high, be more conservative
        elif mean_kl > self.kl_target * 2.0:
            self.clip_range = max(0.1, self.clip_range * 0.95)
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current hyperparameter values."""
        return {
            'learning_rate': self.learning_rate,
            'entropy_coef': self.entropy_coef,
            'clip_range': self.clip_range,
            'kl_coef': self.config['training'].get('kl_coef', 0.0),
        }
    
    def should_early_stop(self) -> bool:
        """
        Determine if training should early stop due to divergence.
        Implements Coconut's stability guard.
        """
        if len(self.loss_history) < 20:
            return False
        
        # Check for loss explosion
        recent_loss = np.mean(list(self.loss_history)[-5:])
        old_loss = np.mean(list(self.loss_history)[:5])
        
        if recent_loss > old_loss * 10:
            warnings.warn("Loss explosion detected - consider early stopping")
            return True
        
        # Check for reward collapse
        if len(self.reward_history) >= 50:
            recent_reward = np.mean(list(self.reward_history)[-10:])
            if recent_reward < self.best_reward * 0.5:
                warnings.warn("Reward collapse detected - consider early stopping")
                return True
        
        return False


class CurriculumManager:
    """
    Manages curriculum learning stages.
    Inspired by Ripple's adaptive rollout length and progressive difficulty.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration with curriculum stages
        """
        self.config = config
        self.enabled = config.get('curriculum', {}).get('enabled', False)
        
        if self.enabled:
            self.stages = config['curriculum']['stages']
            self.current_stage_idx = 0
            self.current_stage = self.stages[0]
        else:
            self.stages = []
            self.current_stage_idx = 0
            self.current_stage = None
    
    def update(self, timesteps: int) -> Optional[Dict[str, Any]]:
        """
        Update curriculum stage based on training progress.
        
        Args:
            timesteps: Current training timesteps
            
        Returns:
            New stage config if stage changed, None otherwise
        """
        if not self.enabled:
            return None
        
        # Check if should advance to next stage
        if self.current_stage_idx < len(self.stages) - 1:
            next_stage = self.stages[self.current_stage_idx + 1]
            if timesteps >= next_stage['timesteps']:
                self.current_stage_idx += 1
                self.current_stage = next_stage
                return self.current_stage
        
        return None
    
    def get_current_stage(self) -> Optional[Dict[str, Any]]:
        """Get current curriculum stage."""
        return self.current_stage
    
    def get_stage_name(self) -> str:
        """Get current stage name."""
        if self.current_stage:
            return self.current_stage['name']
        return "default"


class AdaptiveRolloutLength:
    """
    Ripple-inspired adaptive rollout length.
    Use shorter episodes early in training, longer as skill increases.
    """
    
    def __init__(
        self,
        min_length: int = 512,
        max_length: int = 4096,
        initial_length: int = 1024
    ):
        """
        Args:
            min_length: Minimum rollout length
            max_length: Maximum rollout length
            initial_length: Starting rollout length
        """
        self.min_length = min_length
        self.max_length = max_length
        self.current_length = initial_length
        
        self.performance_history = deque(maxlen=50)
        self.best_performance = float('-inf')
    
    def update(self, mean_reward: float) -> int:
        """
        Update rollout length based on performance.
        
        Args:
            mean_reward: Mean episode reward
            
        Returns:
            New rollout length
        """
        self.performance_history.append(mean_reward)
        
        if len(self.performance_history) < 10:
            return self.current_length
        
        recent_perf = np.mean(list(self.performance_history)[-10:])
        
        # Increase rollout length as performance improves
        if recent_perf > self.best_performance:
            self.best_performance = recent_perf
            # Gradually increase rollout length
            self.current_length = min(
                self.max_length,
                int(self.current_length * 1.1)
            )
        
        return self.current_length
    
    def get_length(self) -> int:
        """Get current rollout length."""
        return self.current_length


class RewardMixer:
    """
    Tenshi-inspired reward mixer.
    Dynamically balance goal-based, positional, and aesthetic rewards.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Reward configuration
        """
        self.config = config
        self.weights = {
            'goal': 1.0,
            'positional': 0.5,
            'aesthetic': 0.3,
            'mechanical': 0.4
        }
        
        self.performance_by_type = {k: deque(maxlen=100) for k in self.weights.keys()}
    
    def update_weights(self, rewards_by_type: Dict[str, float]):
        """
        Update reward component weights based on their effectiveness.
        
        Args:
            rewards_by_type: Dictionary of reward contributions by type
        """
        for reward_type, reward in rewards_by_type.items():
            if reward_type in self.performance_by_type:
                self.performance_by_type[reward_type].append(reward)
        
        # Adjust weights based on variance and impact
        for reward_type in self.weights.keys():
            if len(self.performance_by_type[reward_type]) >= 20:
                perf = list(self.performance_by_type[reward_type])
                mean_perf = np.mean(perf)
                
                # Increase weight for effective rewards
                if mean_perf > 0:
                    self.weights[reward_type] *= 1.01
                else:
                    self.weights[reward_type] *= 0.99
                
                # Normalize to sum to reasonable value
                total = sum(self.weights.values())
                if total > 5.0:
                    for k in self.weights:
                        self.weights[k] /= (total / 3.0)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current reward weights."""
        return self.weights.copy()
