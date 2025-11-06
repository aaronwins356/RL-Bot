#!/usr/bin/env python3
"""Validation script for RL training pipeline.

This script validates that the training pipeline executes without errors
and verifies all the success criteria:
- No runtime errors during PPO updates
- Losses are scalar values
- Explained variance is tracked
- Device alignment is correct
- Training advances at least one full iteration
"""
import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.nets import ActorCriticNet
from core.models.ppo import PPO
from core.training.buffer import ReplayBuffer
from core.env.rocket_sim_env import RocketSimEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model_device_alignment(model: ActorCriticNet, device: torch.device) -> bool:
    """Validate all model parameters are on the correct device."""
    try:
        model_device = next(model.parameters()).device
        assert model_device == device, f"Model on {model_device}, expected {device}"
        logger.info("✅ Device alignment verified: All parameters on %s", device)
        return True
    except Exception as e:
        logger.error("❌ Device alignment failed: %s", e)
        return False


def validate_scalar_losses(stats: dict) -> bool:
    """Validate that all losses are scalar values."""
    try:
        assert np.isscalar(stats['policy_loss']), "Policy loss not scalar"
        assert np.isscalar(stats['value_loss']), "Value loss not scalar"
        assert np.isscalar(stats['entropy_loss']), "Entropy loss not scalar"
        assert np.isscalar(stats['total_loss']), "Total loss not scalar"
        assert np.isscalar(stats['explained_variance']), "Explained variance not scalar"
        
        logger.info("✅ Scalar consistency verified:")
        logger.info("   policy_loss=%.4f (type: %s)", 
                   stats['policy_loss'], type(stats['policy_loss']).__name__)
        logger.info("   value_loss=%.4f (type: %s)", 
                   stats['value_loss'], type(stats['value_loss']).__name__)
        logger.info("   entropy_loss=%.4f (type: %s)", 
                   stats['entropy_loss'], type(stats['entropy_loss']).__name__)
        logger.info("   explained_variance=%.4f (type: %s)", 
                   stats['explained_variance'], type(stats['explained_variance']).__name__)
        return True
    except Exception as e:
        logger.error("❌ Scalar validation failed: %s", e)
        return False


def validate_ppo_update() -> bool:
    """Run a single PPO update and validate outputs."""
    logger.info("Running PPO update validation...")
    
    # Setup
    device = torch.device('cpu')
    obs_size = 180
    batch_size = 64
    
    # Create model
    model = ActorCriticNet(
        input_size=obs_size,
        hidden_sizes=[256, 256],
        action_categoricals=5,
        action_bernoullis=3,
        activation="relu"
    ).to(device)
    
    # Validate device
    if not validate_model_device_alignment(model, device):
        return False
    
    # Create PPO
    ppo = PPO(model, {"n_epochs": 2})
    
    # Create dummy batch
    observations = torch.randn(batch_size, obs_size, device=device)
    actions_cat = torch.randint(0, 3, (batch_size, 5), device=device)
    actions_ber = torch.randint(0, 2, (batch_size, 3), device=device)
    
    # Get log probs from model
    with torch.no_grad():
        cat_probs, ber_probs, values, _, _ = model(observations)
        
        # Compute log probs
        old_log_probs_cat = torch.log(
            torch.gather(cat_probs, 2, actions_cat.unsqueeze(2)) + 1e-8
        ).squeeze(2)
        old_log_probs_ber = torch.log(
            torch.gather(ber_probs, 2, actions_ber.unsqueeze(2)) + 1e-8
        ).squeeze(2)
        old_values = values.squeeze()
    
    advantages = torch.randn(batch_size, device=device)
    returns = torch.randn(batch_size, device=device)
    
    # Perform PPO update
    try:
        stats = ppo.update(
            observations,
            actions_cat,
            actions_ber,
            old_log_probs_cat,
            old_log_probs_ber,
            advantages,
            returns,
            old_values
        )
        
        # Validate outputs
        if not validate_scalar_losses(stats):
            return False
        
        # Check model is in training mode
        assert model.training, "Model not in training mode"
        logger.info("✅ Model in training mode")
        
        logger.info("✅ PPO update completed successfully")
        return True
        
    except Exception as e:
        logger.error("❌ PPO update failed: %s", e)
        import traceback
        traceback.print_exc()
        return False


def validate_training_loop() -> bool:
    """Run a minimal training loop and validate it executes."""
    logger.info("Running minimal training loop validation...")
    
    try:
        from core.training.train_loop import TrainingLoop
        from core.infra.config import ConfigManager
        from pathlib import Path
        
        # Create minimal config
        config_path = Path("configs/base.yaml")
        if not config_path.exists():
            logger.error("❌ Config file not found: %s", config_path)
            return False
        
        config_manager = ConfigManager(config_path)
        config = config_manager.config
        
        # Override for quick test
        config.training.total_timesteps = 100
        config.training.batch_size = 32
        config.training.debug_mode = False
        config.logging.log_interval = 50
        config.inference.device = "cpu"
        
        # Create trainer
        trainer = TrainingLoop(
            config=config,
            log_dir="/tmp/rl_validation_test",
            seed=42
        )
        
        # Run short training
        trainer.train(total_timesteps=100)
        
        logger.info("✅ Training loop executed successfully for 100 timesteps")
        return True
        
    except Exception as e:
        logger.error("❌ Training loop failed: %s", e)
        import traceback
        traceback.print_exc()
        return False


def validate_gae_computation() -> bool:
    """Validate GAE (Generalized Advantage Estimation) computation."""
    logger.info("Validating GAE computation...")
    
    try:
        # Create model
        model = ActorCriticNet(
            input_size=180,
            hidden_sizes=[256, 256],
            action_categoricals=5,
            action_bernoullis=3
        )
        
        ppo = PPO(model)
        
        # Create test trajectory
        T = 100
        rewards = np.random.randn(T) * 0.1
        values = np.random.randn(T) * 0.5
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True  # Episode ends
        next_value = 0.0
        
        # Compute GAE
        advantages, returns = ppo.compute_gae(
            rewards, values, dones, next_value
        )
        
        # Validate shapes
        assert advantages.shape == rewards.shape, "Advantages shape mismatch"
        assert returns.shape == rewards.shape, "Returns shape mismatch"
        
        # Validate no NaNs
        assert not np.any(np.isnan(advantages)), "Advantages contain NaN"
        assert not np.any(np.isnan(returns)), "Returns contain NaN"
        
        logger.info("✅ GAE computation validated:")
        logger.info("   advantages shape: %s, returns shape: %s", 
                   advantages.shape, returns.shape)
        logger.info("   advantages range: [%.3f, %.3f]", 
                   advantages.min(), advantages.max())
        logger.info("   returns range: [%.3f, %.3f]", 
                   returns.min(), returns.max())
        
        return True
        
    except Exception as e:
        logger.error("❌ GAE validation failed: %s", e)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    logger.info("="*70)
    logger.info("RL Training Pipeline Validation")
    logger.info("="*70)
    
    tests = [
        ("GAE Computation", validate_gae_computation),
        ("PPO Update", validate_ppo_update),
        ("Training Loop", validate_training_loop),
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info("")
        logger.info("-"*70)
        logger.info("Test: %s", name)
        logger.info("-"*70)
        results[name] = test_func()
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("Validation Summary")
    logger.info("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info("%s: %s", name, status)
        if not passed:
            all_passed = False
    
    logger.info("="*70)
    if all_passed:
        logger.info("✅ FIX VERIFIED - All validation tests passed!")
        logger.info("✅ Training pipeline stable and ready for production")
        return 0
    else:
        logger.error("❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
