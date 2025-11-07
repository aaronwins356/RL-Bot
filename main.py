"""
Main entry point for RL-Bot training.
Run this script to start training: python main.py
"""

import argparse
import sys
from pathlib import Path

from rl_bot.core.utils import load_config, get_device, setup_logging, set_seed
from rl_bot.core.env_setup import make_vec_env
from rl_bot.train import train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RL-Bot - Rocket League AI Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
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
        default=None,
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training (overrides config)'
    )
    
    parser.add_argument(
        '--num-envs',
        type=int,
        default=None,
        help='Number of parallel environments (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs and TensorBoard'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 60)
    print("RL-Bot - Rocket League AI Training".center(60))
    print("=" * 60)
    print()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"✗ Error: Configuration file not found: {args.config}")
        print(f"  Please create a config.yaml file or specify --config")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.timesteps is not None:
        config['training']['total_timesteps'] = args.timesteps
    if args.device is not None:
        config['device'] = args.device
    if args.num_envs is not None:
        config['environment']['num_envs'] = args.num_envs
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"✓ Random seed set to: {seed}")
    
    # Setup logging
    logger = setup_logging(args.log_dir, verbose=args.verbose or True)
    logger.info("Starting RL-Bot training")
    logger.info(f"Configuration: {args.config}")
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"✓ Device: {device}")
    logger.info(f"Using device: {device}")
    
    # Create environment
    print("\nCreating training environment...")
    try:
        num_envs = config.get('environment', {}).get('num_envs', 4)
        env = make_vec_env(config, num_envs)
        print(f"✓ Created {num_envs} parallel environments")
        logger.info(f"Created {num_envs} parallel environments")
        
        # Print environment info
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        print(f"  Observation dimension: {obs_dim}")
        print(f"  Action dimension: {action_dim}")
        logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
        
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        logger.error(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Training configuration
    total_timesteps = config.get('training', {}).get('total_timesteps', 10_000_000)
    batch_size = config.get('training', {}).get('batch_size', 4096)
    n_steps = config.get('training', {}).get('n_steps', 2048)
    
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Steps per update: {n_steps:,}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Log directory: {args.log_dir}")
    
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        logger.info(f"Resuming training from checkpoint: {args.resume}")
    
    logger.info(f"Training for {total_timesteps:,} timesteps")
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting Training".center(60))
    print("=" * 60)
    print()
    
    try:
        model = train(
            env=env,
            config=config,
            total_timesteps=total_timesteps,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            logger=logger,
            resume_from=args.resume
        )
        
        print("\n" + "=" * 60)
        print("Training Complete!".center(60))
        print("=" * 60)
        print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
        print(f"Logs saved to: {args.log_dir}")
        print(f"\nView training progress with TensorBoard:")
        print(f"  tensorboard --logdir {args.log_dir}")
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        logger.info("Training interrupted by user")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up
        env.close()
        logger.info("Environment closed")
    
    print("\nThank you for using RL-Bot!")


if __name__ == "__main__":
    main()
