"""Hyperparameter tuning using Optuna for RL-Bot.

This module implements automated hyperparameter search:
- Search over LR [8e-5 → 4e-4]
- Entropy coefficient [0.005 → 0.03]  
- Batch size [4096, 8192, 16384]
- Objective: maximize Elo after 300k steps
"""

import optuna
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuner for PPO."""
    
    def __init__(
        self,
        base_config_path: str = "configs/base.yaml",
        n_trials: int = 50,
        target_timesteps: int = 300000,
        results_path: str = "tuning_results.json",
    ):
        """Initialize tuner.
        
        Args:
            base_config_path: Path to base configuration file
            n_trials: Number of optimization trials
            target_timesteps: Training timesteps per trial
            results_path: Path to save tuning results
        """
        self.base_config_path = Path(base_config_path)
        self.n_trials = n_trials
        self.target_timesteps = target_timesteps
        self.results_path = Path(results_path)
        
        # Load base config
        with open(self.base_config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        # Storage for results
        self.trial_results = []
        
        logger.info(f"HyperparameterTuner initialized:")
        logger.info(f"  - Base config: {base_config_path}")
        logger.info(f"  - Trials: {n_trials}")
        logger.info(f"  - Target timesteps: {target_timesteps}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Final Elo rating (objective to maximize)
        """
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 8e-5, 4e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.005, 0.03, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
        
        # Optional: additional hyperparameters
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
        n_epochs = trial.suggest_int("n_epochs", 3, 6)
        
        logger.info(f"Trial {trial.number}: LR={learning_rate:.6f}, "
                   f"ent={ent_coef:.4f}, batch={batch_size}")
        
        # Create trial config
        trial_config = self._create_trial_config(
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            batch_size=batch_size,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
        )
        
        # Run training (placeholder - in real implementation would actually train)
        final_elo = self._run_training(trial_config, trial.number)
        
        # Store result
        self.trial_results.append({
            "trial": trial.number,
            "params": trial.params,
            "elo": final_elo,
        })
        
        logger.info(f"Trial {trial.number} completed: Elo={final_elo:.1f}")
        
        return final_elo
    
    def _create_trial_config(
        self,
        learning_rate: float,
        ent_coef: float,
        batch_size: int,
        clip_range: float,
        gae_lambda: float,
        n_epochs: int,
    ) -> Dict[str, Any]:
        """Create config for trial.
        
        Args:
            learning_rate: Learning rate value
            ent_coef: Entropy coefficient
            batch_size: Batch size
            clip_range: Clip range for PPO
            gae_lambda: GAE lambda value
            n_epochs: Number of epochs
            
        Returns:
            Trial configuration dict
        """
        config = self.base_config.copy()
        
        # Update training parameters
        config["training"]["learning_rate"] = learning_rate
        config["training"]["ent_coef"] = ent_coef
        config["training"]["batch_size"] = batch_size
        config["training"]["clip_range"] = clip_range
        config["training"]["gae_lambda"] = gae_lambda
        config["training"]["n_epochs"] = n_epochs
        config["training"]["total_timesteps"] = self.target_timesteps
        
        # Disable evaluation during tuning for speed
        if "evaluation" in config.get("training", {}):
            config["training"]["evaluation"]["enabled"] = False
        
        # Reduce logging frequency
        config["logging"]["log_interval"] = 10000
        config["logging"]["save_interval"] = 100000  # Don't save during tuning
        
        return config
    
    def _run_training(self, config: Dict[str, Any], trial_number: int) -> float:
        """Run training with given config.
        
        This is a placeholder. In real implementation, would:
        1. Create training loop with config
        2. Train for target_timesteps
        3. Evaluate final Elo
        4. Return Elo
        
        Args:
            config: Training configuration
            trial_number: Current trial number
            
        Returns:
            Final Elo rating
        """
        # Placeholder: simulate training with improving Elo
        # In real implementation, would actually train the model
        
        # Simulation constants (optimal hyperparameters for heuristic)
        OPTIMAL_LR = 2e-4
        OPTIMAL_ENT = 0.015
        OPTIMAL_BATCH = 8192
        
        # Simulate based on hyperparameters
        lr = config["training"]["learning_rate"]
        ent = config["training"]["ent_coef"]
        batch = config["training"]["batch_size"]
        
        # Simple heuristic: better performance with moderate values
        lr_score = 1.0 - abs(lr - OPTIMAL_LR) / OPTIMAL_LR
        ent_score = 1.0 - abs(ent - OPTIMAL_ENT) / OPTIMAL_ENT
        batch_score = 1.0 if batch == OPTIMAL_BATCH else 0.8
        
        # Combine scores
        combined_score = (lr_score + ent_score + batch_score) / 3.0
        
        # Map to Elo (1400-1700 range)
        base_elo = 1400
        max_improvement = 300
        final_elo = base_elo + combined_score * max_improvement
        
        logger.info(f"  [Simulated] Trained for {self.target_timesteps} steps")
        logger.info(f"  [Simulated] Final Elo: {final_elo:.1f}")
        
        return final_elo
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Returns:
            Best hyperparameters and results
        """
        logger.info("=" * 60)
        logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
        
        # Create study
        study = optuna.create_study(
            direction="maximize",  # Maximize Elo
            study_name="rlbot_ppo_tuning",
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_elo = best_trial.value
        
        logger.info("=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best Elo: {best_elo:.1f}")
        logger.info("Best hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Save results
        results = {
            "best_params": best_params,
            "best_elo": best_elo,
            "n_trials": self.n_trials,
            "target_timesteps": self.target_timesteps,
            "all_trials": self.trial_results,
        }
        
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.results_path}")
        
        return results
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history.
        
        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.trial_results:
                logger.warning("No trial results to plot")
                return
            
            trials = [r["trial"] for r in self.trial_results]
            elos = [r["elo"] for r in self.trial_results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(trials, elos, 'o-', linewidth=2, markersize=6)
            plt.xlabel("Trial Number", fontsize=12)
            plt.ylabel("Final Elo", fontsize=12)
            plt.title("Hyperparameter Optimization Progress", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")


def main():
    """Main entry point for hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RL-Bot")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300000,
        help="Training timesteps per trial",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuning_results.json",
        help="Path to save results",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create tuner
    tuner = HyperparameterTuner(
        base_config_path=args.config,
        n_trials=args.trials,
        target_timesteps=args.timesteps,
        results_path=args.output,
    )
    
    # Run optimization
    results = tuner.optimize()
    
    # Plot results
    plot_path = Path(args.output).parent / "optimization_history.png"
    tuner.plot_optimization_history(str(plot_path))
    
    logger.info("Hyperparameter tuning complete!")
    logger.info(f"Best Elo: {results['best_elo']:.1f}")
    logger.info("Run training with best parameters to verify results.")


if __name__ == "__main__":
    main()
