#!/usr/bin/env python3
"""Training script for RL-Bot.

This script provides a convenient CLI for training the bot with various configurations.
"""
import argparse
import sys
import json
import logging
import logging.handlers
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.infra.config import ConfigManager
from core.training.train_loop import TrainingLoop

# Try to import colorama for colored output
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback no-op
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


def setup_colored_logging(debug: bool = False):
    """Setup colored logging.
    
    Args:
        debug: Enable debug level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    class ColoredFormatter(logging.Formatter):
        """Colored log formatter."""
        
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT,
        }
        
        def format(self, record):
            if HAS_COLOR:
                levelname = record.levelname
                record.levelname = f"{self.COLORS.get(levelname, '')}{levelname}{Style.RESET_ALL}"
            return super().format(record)
    
    # Setup handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


def get_git_info() -> dict:
    """Get git commit hash and status.
    
    Returns:
        Dictionary with git info
    """
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Check if there are uncommitted changes
        dirty = subprocess.call(
            ['git', 'diff-index', '--quiet', 'HEAD', '--'],
            stderr=subprocess.DEVNULL
        ) != 0
        
        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'dirty': dirty
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'commit_hash': 'unknown',
            'branch': 'unknown',
            'dirty': False
        }


def save_run_metadata(log_dir: Path, config_dict: dict, args: argparse.Namespace):
    """Save run metadata for reproducibility.
    
    Args:
        log_dir: Log directory
        config_dict: Configuration dictionary
        args: Command line arguments
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config_dict,
        'git': get_git_info(),
        'cli_args': vars(args)
    }
    
    metadata_path = log_dir / 'run_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved run metadata to {metadata_path}")


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
        default=None,
        help='Directory for logs and checkpoints (overrides config)'
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
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (1 episode / 1k steps with verbose output)'
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_colored_logging(debug=args.debug)
    
    logger.info("=" * 70)
    logger.info("RL-Bot Training")
    logger.info("=" * 70)
    
    # Load configuration with ConfigManager
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        config_manager = ConfigManager(config_path)
        config_manager.validate_schema()
        logger.info("Configuration schema validated successfully")
    except Exception as e:
        logger.error(f"Failed to load or validate config: {e}")
        sys.exit(1)
    
    # Build overrides dictionary
    overrides = {}
    
    if args.timesteps is not None:
        overrides.setdefault('training', {})['total_timesteps'] = args.timesteps
        logger.info(f"Overriding total timesteps: {args.timesteps}")
    
    if args.device is not None:
        overrides.setdefault('inference', {})['device'] = args.device
        logger.info(f"Overriding device: {args.device}")
    
    if args.logdir is not None:
        overrides.setdefault('logging', {})['log_dir'] = args.logdir
        logger.info(f"Overriding log directory: {args.logdir}")
    
    if args.eval_freq is not None:
        overrides.setdefault('logging', {})['eval_interval'] = args.eval_freq
        logger.info(f"Overriding eval frequency: {args.eval_freq}")
    
    if args.save_freq is not None:
        overrides.setdefault('logging', {})['save_interval'] = args.save_freq
        logger.info(f"Overriding save frequency: {args.save_freq}")
    
    if args.seed is not None:
        overrides.setdefault('training', {})['seed'] = args.seed
        logger.info(f"Using random seed: {args.seed}")
    
    if args.aerial_curriculum:
        logger.info("Enabling aerial-focused curriculum training")
        overrides.setdefault('training', {}).setdefault('curriculum', {})['aerial_focus'] = True
    
    if args.offline_pretrain:
        logger.info("Enabling offline pretraining")
        overrides.setdefault('training', {}).setdefault('offline', {})['enabled'] = True
    
    if args.debug:
        logger.warning("Running in DEBUG mode - 1k steps only with verbose output")
        overrides.setdefault('training', {})['total_timesteps'] = 1000
        overrides.setdefault('logging', {})['log_interval'] = 10
    
    # Apply overrides
    if overrides:
        config_manager.apply_overrides(overrides)
    
    config = config_manager.config
    
    # Get log directory
    log_dir = Path(config.get('logging', {}).get('log_dir', args.logdir or 'logs'))
    
    # Save run metadata
    save_run_metadata(log_dir, config.to_dict(), args)
    
    # Print configuration summary
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Algorithm: {config.training.algorithm}")
    logger.info(f"  Total timesteps: {config.training.total_timesteps:,}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Device: {config.inference.device}")
    logger.info(f"  Log directory: {log_dir}")
    
    # Show offline pretraining status
    if config_manager.get_safe('training.offline.enabled', False):
        dataset_path = config_manager.get_safe('training.offline.dataset_path', 'N/A')
        pretrain_epochs = config_manager.get_safe('training.offline.pretrain_epochs', 10)
        logger.info(f"  Offline pretraining: Enabled")
        logger.info(f"    Dataset: {dataset_path}")
        logger.info(f"    Epochs: {pretrain_epochs}")
    
    # Show curriculum status
    if config_manager.get_safe('training.curriculum.aerial_focus', False):
        logger.info(f"  Aerial curriculum: Enabled")
    
    logger.info("")
    
    # Setup rotating file logging
    file_log_path = log_dir / 'train.log'
    file_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        file_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"File logging enabled: {file_log_path}")
    
    # Create training loop
    try:
        trainer = TrainingLoop(
            config=config,
            log_dir=str(log_dir),
            checkpoint_path=args.checkpoint,
            seed=args.seed
        )
        
        logger.info("Starting training...")
        logger.info("")
        
        # Run training
        trainer.train(total_timesteps=config.training.total_timesteps)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training completed successfully!")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("Training interrupted by user")
        logger.warning("Saving final checkpoint...")
        logger.warning("=" * 70)
        # Trainer will auto-save on interruption
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"Training failed with error: {e}")
        logger.error("=" * 70)
        if args.debug:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
