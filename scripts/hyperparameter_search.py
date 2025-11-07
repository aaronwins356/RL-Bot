#!/usr/bin/env python3
"""Hyperparameter optimization script using Optuna.

This script performs automated hyperparameter search to find optimal configurations
for achieving Elo 1550-1700 within 150k-1M timesteps.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

import torch
import numpy as np

from core.infra.config import ConfigManager
from core.training.train_loop import TrainingLoop

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config_path: Path, target_timesteps: int = 300000) -> float:
    """Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        base_config_path: Path to base configuration file
        target_timesteps: Number of timesteps to train for evaluation
        
    Returns:
        Final Elo rating (objective to maximize)
    """
    # Load base configuration
    config_manager = ConfigManager(base_config_path)
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 12288, 16384, 24576])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.02, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    # Network architecture
    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [256, 512, 768, 1024])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [256, 512, 768])
    hidden_size_3 = trial.suggest_categorical("hidden_size_3", [128, 256, 512])
    
    # Normalization
    clip_obs = trial.suggest_float("clip_obs", 5.0, 20.0)
    clip_reward = trial.suggest_float("clip_reward", 5.0, 20.0)
    
    # Apply overrides to config
    overrides = {
        "training": {
            "total_timesteps": target_timesteps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "normalization": {
                "normalize_observations": True,
                "normalize_rewards": True,
                "clip_obs": clip_obs,
                "clip_reward": clip_reward,
            }
        },
        "network": {
            "hidden_sizes": [hidden_size_1, hidden_size_2, hidden_size_3]
        },
        "logging": {
            "tensorboard": False,  # Disable for faster training
            "log_interval": 5000,
            "save_interval": 50000,
            "eval_interval": 50000,
            "eval_num_games": 5,  # Fewer games for faster eval
        }
    }
    
    config_manager.apply_overrides(overrides)
    config = config_manager.config
    
    # Create log directory for this trial
    log_dir = Path(f"logs/optuna_trial_{trial.number}")
    
    try:
        # Initialize training
        trainer = TrainingLoop(
            config=config,
            log_dir=str(log_dir),
            seed=trial.number,  # Use trial number as seed for reproducibility
            auto_resume=False,
        )
        
        # Train for target timesteps
        trainer.train(total_timesteps=target_timesteps)
        
        # Get final Elo rating
        final_elo = trainer.evaluator.get_elo()
        
        # Report intermediate values for pruning
        trial.report(final_elo, step=target_timesteps)
        
        logger.info(f"Trial {trial.number} completed with Elo: {final_elo:.1f}")
        
        return final_elo
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        # Return very low score for failed trials
        return -1000.0


def run_optimization(
    base_config_path: Path,
    n_trials: int = 50,
    target_timesteps: int = 300000,
    n_jobs: int = 1,
    study_name: str = "rlbot_optimization",
    storage: str = None,
) -> Dict[str, Any]:
    """Run hyperparameter optimization with Optuna.
    
    Args:
        base_config_path: Path to base configuration file
        n_trials: Number of trials to run
        target_timesteps: Timesteps per trial
        n_jobs: Number of parallel jobs (1 = sequential)
        study_name: Name for the study
        storage: Database URL for distributed optimization (e.g., "sqlite:///optuna.db")
        
    Returns:
        Dictionary with best parameters and results
    """
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize Elo
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=100000),
        storage=storage,
        load_if_exists=True,
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config_path, target_timesteps),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info("=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Elo: {best_value:.1f}")
    logger.info(f"Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Save results
    results = {
        "best_params": best_params,
        "best_elo": best_value,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }
    
    results_path = Path("logs") / f"{study_name}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Create optimized config file
    config_manager = ConfigManager(base_config_path)
    overrides = {
        "training": {
            "learning_rate": best_params["learning_rate"],
            "batch_size": best_params["batch_size"],
            "n_epochs": best_params["n_epochs"],
            "gae_lambda": best_params["gae_lambda"],
            "clip_range": best_params["clip_range"],
            "ent_coef": best_params["ent_coef"],
            "vf_coef": best_params["vf_coef"],
            "max_grad_norm": best_params["max_grad_norm"],
            "normalization": {
                "clip_obs": best_params["clip_obs"],
                "clip_reward": best_params["clip_reward"],
            }
        },
        "network": {
            "hidden_sizes": [
                best_params["hidden_size_1"],
                best_params["hidden_size_2"],
                best_params["hidden_size_3"]
            ]
        }
    }
    
    config_manager.apply_overrides(overrides)
    
    # Save optimized config
    optimized_config_path = Path("configs") / f"{study_name}_best.yaml"
    config_manager.save_to_file(optimized_config_path)
    logger.info(f"Optimized config saved to: {optimized_config_path}")
    
    return results


def main():
    """Main entry point for hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for RL-Bot using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_optimized.yaml",
        help="Base configuration file"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    
    parser.add_argument(
        "--target-timesteps",
        type=int,
        default=300000,
        help="Timesteps to train each trial (recommend 150k-300k)"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (1 = sequential, >1 = parallel)"
    )
    
    parser.add_argument(
        "--study-name",
        type=str,
        default="rlbot_optimization",
        help="Name for the optimization study"
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for distributed optimization (e.g., sqlite:///optuna.db)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 70)
    logger.info("RL-Bot Hyperparameter Optimization with Optuna")
    logger.info("=" * 70)
    logger.info(f"Base config: {args.config}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Target timesteps per trial: {args.target_timesteps}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info(f"Study name: {args.study_name}")
    logger.info("=" * 70)
    
    # Verify CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, training will be slow")
    
    # Run optimization
    results = run_optimization(
        base_config_path=Path(args.config),
        n_trials=args.n_trials,
        target_timesteps=args.target_timesteps,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        storage=args.storage,
    )
    
    logger.info("=" * 70)
    logger.info("Optimization complete!")
    logger.info(f"Best Elo achieved: {results['best_elo']:.1f}")
    logger.info("See logs/ directory for detailed results and optimized config")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
