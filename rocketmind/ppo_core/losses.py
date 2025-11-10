"""
PPO loss functions with clipping and entropy bonus.
Includes KL divergence monitoring and adaptive clipping.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    normalize_advantage: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO clipped surrogate loss.
    
    Args:
        log_probs: New log probabilities
        old_log_probs: Old log probabilities
        advantages: Advantage estimates
        clip_range: Clipping range
        normalize_advantage: Whether to normalize advantages
        
    Returns:
        loss: PPO loss
        clip_fraction: Fraction of clipped ratios
        approx_kl: Approximate KL divergence
    """
    # Normalize advantages
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    # Compute metrics
    with torch.no_grad():
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
        approx_kl = torch.mean(old_log_probs - log_probs)
    
    return policy_loss, clip_fraction, approx_kl


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_range: float = 0.2,
    use_clipping: bool = False
) -> torch.Tensor:
    """
    Compute value function loss.
    
    Args:
        values: Predicted values
        returns: Target returns
        old_values: Old value predictions (for clipping)
        clip_range: Clipping range
        use_clipping: Whether to clip value loss
        
    Returns:
        loss: Value loss
    """
    if use_clipping and old_values is not None:
        # Clipped value loss (helps stability)
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -clip_range,
            clip_range
        )
        loss_unclipped = F.mse_loss(values, returns)
        loss_clipped = F.mse_loss(values_clipped, returns)
        return torch.max(loss_unclipped, loss_clipped)
    else:
        # Standard MSE loss
        return F.mse_loss(values, returns)


def entropy_bonus(entropy: torch.Tensor, ent_coef: float = 0.01) -> torch.Tensor:
    """
    Compute entropy bonus for exploration.
    
    Args:
        entropy: Policy entropy
        ent_coef: Entropy coefficient
        
    Returns:
        bonus: Entropy bonus (negative because we maximize entropy)
    """
    return -ent_coef * entropy.mean()


def total_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    normalize_advantage: bool = True,
    use_value_clipping: bool = False
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total PPO loss (policy + value + entropy).
    
    Args:
        log_probs: New log probabilities
        old_log_probs: Old log probabilities
        advantages: Advantage estimates
        values: Predicted values
        returns: Target returns
        entropy: Policy entropy
        old_values: Old value predictions
        clip_range: Clipping range
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        normalize_advantage: Whether to normalize advantages
        use_value_clipping: Whether to clip value loss
        
    Returns:
        total_loss: Combined loss
        info: Dictionary with loss components and metrics
    """
    # Policy loss
    policy_loss, clip_fraction, approx_kl = ppo_loss(
        log_probs,
        old_log_probs,
        advantages,
        clip_range,
        normalize_advantage
    )
    
    # Value loss
    v_loss = value_loss(
        values,
        returns,
        old_values,
        clip_range,
        use_value_clipping
    )
    
    # Entropy bonus
    ent_loss = entropy_bonus(entropy, ent_coef)
    
    # Total loss
    total_loss = policy_loss + vf_coef * v_loss + ent_loss
    
    # Info dictionary
    info = {
        'policy_loss': policy_loss.item(),
        'value_loss': v_loss.item(),
        'entropy_loss': ent_loss.item(),
        'total_loss': total_loss.item(),
        'clip_fraction': clip_fraction.item(),
        'approx_kl': approx_kl.item(),
        'entropy': entropy.mean().item()
    }
    
    return total_loss, info


def adaptive_clip_range(
    approx_kl: float,
    target_kl: float = 0.02,
    clip_range: float = 0.2,
    adaptation_rate: float = 0.1
) -> float:
    """
    Adaptively adjust clip range based on KL divergence.
    
    Args:
        approx_kl: Approximate KL divergence
        target_kl: Target KL divergence
        clip_range: Current clip range
        adaptation_rate: How fast to adapt
        
    Returns:
        new_clip_range: Adjusted clip range
    """
    if approx_kl > target_kl * 1.5:
        # KL too high, reduce clip range
        new_clip_range = clip_range * (1.0 - adaptation_rate)
    elif approx_kl < target_kl * 0.5:
        # KL too low, increase clip range
        new_clip_range = clip_range * (1.0 + adaptation_rate)
    else:
        new_clip_range = clip_range
    
    # Clamp to reasonable range
    return max(0.05, min(0.5, new_clip_range))


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute explained variance (R²-like metric).
    
    Args:
        y_pred: Predictions
        y_true: True values
        
    Returns:
        explained_var: Explained variance (1.0 is perfect, 0.0 is no better than mean)
    """
    var_y = torch.var(y_true)
    if var_y == 0:
        return 0.0
    return 1.0 - torch.var(y_true - y_pred) / var_y


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Rewards [T, N]
        values: Value predictions [T, N]
        dones: Done flags [T, N]
        gamma: Discount factor
        gae_lambda: GAE lambda
        
    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    last_gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
            next_non_terminal = 0
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    
    return advantages, returns
