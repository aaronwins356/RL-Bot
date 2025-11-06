#!/usr/bin/env python3
"""Training script for RL-Bot.

This script provides a convenient CLI for training the bot with various configurations.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.infra.config import load_config
from core.training.train_loop import TrainingLoop


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RL-Bot with PPO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Total training timesteps (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device for training (overrides config)'
    )
    
    parser.add_argument(
        '--logdir',
        type=str,
        default='logs',
        help='Directory for logs and checkpoints'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=None,
        help='Evaluate every N timesteps (overrides config)'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=None,
        help='Save checkpoint every N timesteps (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--aerial-curriculum',
        action='store_true',
        help='Use aerial-focused curriculum training'
    )
    
    parser.add_argument(
        '--offline-pretrain',
        action='store_true',
        help='Pretrain with offline dataset before RL training'
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("RL-Bot Training")
    print("=" * 70)
    print()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Apply command line overrides
    if args.timesteps is not None:
        config.training['total_timesteps'] = args.timesteps
        print(f"Overriding total timesteps: {args.timesteps}")
    
    if args.device is not None:
        config.inference['device'] = args.device
        print(f"Overriding device: {args.device}")
    
    if args.eval_freq is not None:
        config.logging['eval_interval'] = args.eval_freq
        print(f"Overriding eval frequency: {args.eval_freq}")
    
    if args.save_freq is not None:
        config.logging['save_interval'] = args.save_freq
        print(f"Overriding save frequency: {args.save_freq}")
    
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")
    
    if args.aerial_curriculum:
        print("Enabling aerial-focused curriculum training")
        # Would modify config to enable aerial curriculum
    
    if args.offline_pretrain:
        print("Enabling offline pretraining")
        config.training['offline']['enabled'] = True
    
    print()
    print("Training Configuration:")
    print(f"  Algorithm: {config.training['algorithm']}")
    print(f"  Total timesteps: {config.training['total_timesteps']:,}")
    print(f"  Batch size: {config.training['batch_size']}")
    print(f"  Learning rate: {config.training['learning_rate']}")
    print(f"  Device: {config.inference['device']}")
    print(f"  Log directory: {args.logdir}")
    print()
    
    # Create training loop
    try:
        trainer = TrainingLoop(
            config=config,
            log_dir=args.logdir,
            checkpoint_path=args.checkpoint,
            seed=args.seed
        )
        
        print("Starting training...")
        print()
        
        # Run training
        trainer.train(total_timesteps=config.training['total_timesteps'])
        
        print()
        print("=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Training interrupted by user")
        print("Saving final checkpoint...")
        print("=" * 70)
        # Trainer will auto-save on interruption
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"Training failed with error: {e}")
        print("=" * 70)
        raise


if __name__ == '__main__':
    main()
