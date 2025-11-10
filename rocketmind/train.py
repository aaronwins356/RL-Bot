"""
Training script for RocketMind PPO bot.
Main training loop with environment integration.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

# Import from existing rl_bot for environment setup
import sys
sys.path.append(str(Path(__file__).parent.parent))
from rl_bot.core.env_setup import make_vec_env
from rl_bot.core.utils import load_config as load_old_config

from .ppo_core import (
    create_actor_critic,
    RolloutBuffer,
    PPOTrainer,
    get_device,
    set_seed,
    load_config
)


def train(
    config_path: str = "rocketmind/configs/default.yaml",
    checkpoint_dir: str = "checkpoints/rocketmind",
    log_dir: str = "logs/rocketmind",
    resume_from: Optional[str] = None
):
    """
    Main training function for RocketMind.
    
    Args:
        config_path: Path to configuration file
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs and TensorBoard
        resume_from: Optional checkpoint to resume from
    """
    print("=" * 70)
    print("RocketMind PPO Training".center(70))
    print("=" * 70)
    print()
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"✓ Loaded configuration: {config_path}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        print("  Using default rl_bot config instead")
        config = load_old_config("config.yaml")
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"✓ Random seed: {seed}")
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"✓ Device: {device}")
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print("\nCreating environment...")
    num_envs = config.get('environment', {}).get('num_envs', 4)
    env = make_vec_env(config, num_envs)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"✓ Environment created:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Parallel envs: {num_envs}")
    
    # Create model
    print("\nCreating PPO model...")
    model = create_actor_critic(obs_dim, action_dim, config)
    print(f"✓ Model created")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    print(f"✓ TensorBoard logging: {log_dir}")
    
    # Create trainer
    trainer = PPOTrainer(model, config, device, writer)
    
    # Resume from checkpoint if provided
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        trainer.load(resume_from)
        print("✓ Checkpoint loaded")
    
    # Training hyperparameters
    train_config = config.get('training', {})
    total_timesteps = train_config.get('total_timesteps', 10_000_000)
    n_steps = train_config.get('n_steps', 2048)
    batch_size = train_config.get('batch_size', 4096)
    gamma = train_config.get('gamma', 0.99)
    gae_lambda = train_config.get('gae_lambda', 0.95)
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_dim=obs_dim,
        num_envs=num_envs,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda
    )
    
    print("\n" + "=" * 70)
    print("Training Configuration".center(70))
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Steps per update: {n_steps:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Updates: {total_timesteps // (n_steps * num_envs):,}")
    print("=" * 70)
    print()
    
    # Training loop
    obs = env.reset()
    episode_rewards = []
    episode_lengths = []
    
    num_updates = 0
    timesteps = 0
    
    progress_bar = tqdm(total=total_timesteps, desc="Training", unit="steps")
    
    try:
        while timesteps < total_timesteps:
            # Collect rollouts
            for step in range(n_steps):
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
                    action, log_prob, _, value = model.get_action_and_value(obs_tensor)
                
                # Environment step
                action_np = action.cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)
                
                # Store transition
                buffer.add(
                    obs,
                    action_np,
                    reward,
                    done,
                    value.cpu().numpy(),
                    log_prob.cpu().numpy()
                )
                
                obs = next_obs
                timesteps += num_envs
                progress_bar.update(num_envs)
                
                # Track episode statistics
                for i, d in enumerate(done):
                    if d:
                        if 'episode' in info[i]:
                            episode_rewards.append(info[i]['episode']['r'])
                            episode_lengths.append(info[i]['episode']['l'])
            
            # Compute returns and advantages
            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
                _, _, _, next_value = model.get_action_and_value(next_obs_tensor)
            
            buffer.compute_returns_and_advantages(
                next_value.cpu().numpy(),
                np.zeros(num_envs, dtype=np.float32)  # Simplified
            )
            
            # Training step
            progress = timesteps / total_timesteps
            metrics = trainer.train_step(buffer, progress)
            
            # Reset buffer
            buffer.reset()
            
            # Logging
            num_updates += 1
            trainer.total_timesteps = timesteps
            
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards[-100:])
                mean_length = np.mean(episode_lengths[-100:])
                
                writer.add_scalar('rollout/mean_reward', mean_reward, timesteps)
                writer.add_scalar('rollout/mean_length', mean_length, timesteps)
                
                progress_bar.set_postfix({
                    'reward': f'{mean_reward:.2f}',
                    'loss': f'{metrics["total_loss"]:.4f}'
                })
            
            # Save checkpoint
            save_interval = config.get('logging', {}).get('save_interval', 500)
            if num_updates % save_interval == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{timesteps}.pt"
                trainer.save(str(checkpoint_path), metadata={'episode_rewards': episode_rewards})
                print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_path = Path(checkpoint_dir) / "final_model.pt"
        trainer.save(str(final_path))
        print(f"\n✓ Final model saved: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        
    finally:
        progress_bar.close()
        env.close()
        writer.close()
        print("\n✓ Training complete!")
        print(f"  Total timesteps: {timesteps:,}")
        print(f"  Total updates: {num_updates:,}")
        if len(episode_rewards) > 0:
            print(f"  Mean reward: {np.mean(episode_rewards[-100:]):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RocketMind PPO bot")
    parser.add_argument('--config', type=str, default='rocketmind/configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/rocketmind',
                        help='Directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/rocketmind',
                        help='Directory for logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume
    )
