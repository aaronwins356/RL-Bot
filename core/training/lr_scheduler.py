"""Learning rate schedulers for training optimization.

Implements various learning rate scheduling strategies for improved convergence.
"""
import numpy as np
from typing import Optional, Literal


class LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, initial_lr: float):
        """Initialize scheduler.
        
        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step(self, step: int) -> float:
        """Get learning rate for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate
        """
        raise NotImplementedError
    
    def get_lr(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.current_lr


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule.
    
    Gradually decreases learning rate following a cosine curve.
    """
    
    def __init__(
        self,
        initial_lr: float,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
    ):
        """Initialize cosine annealing scheduler.
        
        Args:
            initial_lr: Initial learning rate
            total_steps: Total training steps
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps (linear increase)
        """
        super().__init__(initial_lr)
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
    
    def step(self, step: int) -> float:
        """Get learning rate for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate
        """
        # Warmup phase
        if step < self.warmup_steps:
            self.current_lr = self.initial_lr * (step / self.warmup_steps)
            return self.current_lr
        
        # Cosine annealing phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1 + np.cos(np.pi * progress)
        )
        
        return self.current_lr


class LinearDecayLR(LRScheduler):
    """Linear learning rate decay.
    
    Linearly decreases learning rate from initial to final value.
    """
    
    def __init__(
        self,
        initial_lr: float,
        total_steps: int,
        final_lr_fraction: float = 0.1,
        warmup_steps: int = 0,
    ):
        """Initialize linear decay scheduler.
        
        Args:
            initial_lr: Initial learning rate
            total_steps: Total training steps
            final_lr_fraction: Final LR as fraction of initial (e.g., 0.1 = 10%)
            warmup_steps: Number of warmup steps
        """
        super().__init__(initial_lr)
        self.total_steps = total_steps
        self.final_lr = initial_lr * final_lr_fraction
        self.warmup_steps = warmup_steps
    
    def step(self, step: int) -> float:
        """Get learning rate for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate
        """
        # Warmup phase
        if step < self.warmup_steps:
            self.current_lr = self.initial_lr * (step / self.warmup_steps)
            return self.current_lr
        
        # Linear decay phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        self.current_lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress
        
        return self.current_lr


class ExponentialDecayLR(LRScheduler):
    """Exponential learning rate decay.
    
    Exponentially decreases learning rate.
    """
    
    def __init__(
        self,
        initial_lr: float,
        decay_rate: float = 0.99,
        decay_steps: int = 10000,
    ):
        """Initialize exponential decay scheduler.
        
        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay rate (0-1)
            decay_steps: Apply decay every N steps
        """
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def step(self, step: int) -> float:
        """Get learning rate for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate
        """
        self.current_lr = self.initial_lr * (self.decay_rate ** (step / self.decay_steps))
        return self.current_lr


class AdaptiveLR(LRScheduler):
    """Adaptive learning rate based on performance metrics.
    
    Reduces learning rate when performance plateaus (ReduceLROnPlateau style).
    """
    
    def __init__(
        self,
        initial_lr: float,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        mode: Literal["min", "max"] = "max",
        threshold: float = 1e-4,
    ):
        """Initialize adaptive scheduler.
        
        Args:
            initial_lr: Initial learning rate
            factor: Factor to reduce LR by (new_lr = lr * factor)
            patience: Number of evaluations to wait before reducing LR
            min_lr: Minimum learning rate
            mode: 'min' for loss-like metrics, 'max' for reward-like metrics
            threshold: Minimum change to qualify as improvement
        """
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        
        # State
        self.best_metric = -np.inf if mode == "max" else np.inf
        self.wait = 0
    
    def step(self, step: int, metric: Optional[float] = None) -> float:
        """Get learning rate for current step.
        
        Args:
            step: Current training step (unused but kept for interface compatibility)
            metric: Current performance metric (e.g., Elo, loss)
            
        Returns:
            Learning rate
        """
        if metric is None:
            return self.current_lr
        
        # Check if metric improved
        if self.mode == "max":
            improved = metric > self.best_metric + self.threshold
        else:
            improved = metric < self.best_metric - self.threshold
        
        if improved:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
        
        # Reduce LR if patience exceeded
        if self.wait >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self.wait = 0  # Reset wait counter
        
        return self.current_lr


class ConstantLR(LRScheduler):
    """Constant learning rate (no scheduling)."""
    
    def __init__(self, initial_lr: float):
        """Initialize constant scheduler.
        
        Args:
            initial_lr: Learning rate (constant)
        """
        super().__init__(initial_lr)
    
    def step(self, step: int) -> float:
        """Get learning rate for current step.
        
        Args:
            step: Current training step (unused)
            
        Returns:
            Constant learning rate
        """
        return self.current_lr


def create_lr_scheduler(
    scheduler_type: str,
    initial_lr: float,
    total_steps: int,
    **kwargs
) -> LRScheduler:
    """Factory function to create learning rate scheduler.
    
    Args:
        scheduler_type: Type of scheduler ('cosine', 'linear', 'exponential', 'adaptive', 'constant')
        initial_lr: Initial learning rate
        total_steps: Total training steps
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        LRScheduler instance
    """
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            initial_lr=initial_lr,
            total_steps=total_steps,
            min_lr=kwargs.get("min_lr", initial_lr * 0.1),
            warmup_steps=kwargs.get("warmup_steps", int(total_steps * 0.05)),
        )
    elif scheduler_type == "linear":
        return LinearDecayLR(
            initial_lr=initial_lr,
            total_steps=total_steps,
            final_lr_fraction=kwargs.get("final_lr_fraction", 0.1),
            warmup_steps=kwargs.get("warmup_steps", int(total_steps * 0.05)),
        )
    elif scheduler_type == "exponential":
        return ExponentialDecayLR(
            initial_lr=initial_lr,
            decay_rate=kwargs.get("decay_rate", 0.99),
            decay_steps=kwargs.get("decay_steps", 10000),
        )
    elif scheduler_type == "adaptive":
        return AdaptiveLR(
            initial_lr=initial_lr,
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            min_lr=kwargs.get("min_lr", 1e-6),
            mode=kwargs.get("mode", "max"),
        )
    elif scheduler_type == "constant":
        return ConstantLR(initial_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
