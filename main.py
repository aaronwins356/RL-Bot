"""Main entry point for RL-Bot training and inference.

This module provides a CLI for running training, evaluation, and inference.
"""
import argparse
from pathlib import Path
import sys

from core.infra.config import load_config, Config
from core.training.train_loop import TrainingLoop
from bot_manager import BotManager


def train_command(args):
    """Run training.
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("RL-Bot Training")
    print("=" * 60)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Override with CLI args
    if args.device:
        config.device = args.device
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.logdir:
        config.log_dir = args.logdir
    
    print(f"Config: {config_path}")
    print(f"Device: {config.device}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Log directory: {config.log_dir}")
    print()
    
    # Create training loop
    trainer = TrainingLoop(config, log_dir=args.logdir)
    
    # Run training
    try:
        trainer.train(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving final checkpoint...")
        trainer._save_checkpoint(is_final=True)
    
    print("Training complete!")


def inference_command(args):
    """Run inference (load bot).
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("RL-Bot Inference")
    print("=" * 60)
    
    # Load bot manager
    config_path = Path(args.config) if args.config else None
    model_path = Path(args.model) if args.model else None
    
    bot_manager = BotManager(
        config_path=config_path,
        model_path=model_path,
        policy_type=args.policy
    )
    
    print(f"Policy type: {args.policy}")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print()
    
    # Get stats
    stats = bot_manager.get_stats()
    print("Bot statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nBot loaded successfully!")
    print("Note: To run the bot in RLBot, use the bot.py entry point")


def eval_command(args):
    """Run evaluation.
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("RL-Bot Evaluation")
    print("=" * 60)
    
    from core.training.eval import EloEvaluator
    
    evaluator = EloEvaluator()
    
    # Example evaluation (placeholder)
    print(f"Current Elo: {evaluator.get_elo():.0f}")
    print()
    print("Note: Full evaluation requires running matches in RLBot")
    print("Use the RLBot framework to evaluate against other bots")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RL-Bot: Hybrid Rule-Based + ML Bot for Rocket League",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --config configs/base.yaml --logdir logs/run1
  
  # Load and inspect a trained model
  python main.py inference --model checkpoints/best_model.pt --policy hybrid
  
  # Evaluate model
  python main.py eval --model checkpoints/best_model.pt
  
For running the bot in RLBot, use bot.py as the entry point.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the bot")
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Log directory (overrides config)"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use (overrides config)"
    )
    train_parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)"
    )
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Load and inspect bot")
    inference_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model"
    )
    inference_parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file"
    )
    inference_parser.add_argument(
        "--policy",
        type=str,
        default="hybrid",
        choices=["hybrid", "ml", "rule"],
        help="Policy type to use"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate bot")
    eval_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model"
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Run command
    if args.command == "train":
        train_command(args)
    elif args.command == "inference":
        inference_command(args)
    elif args.command == "eval":
        eval_command(args)


if __name__ == "__main__":
    main()
