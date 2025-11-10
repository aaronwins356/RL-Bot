"""
Main entry point for RocketMind.
CLI interface for training, evaluation, and deployment.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RocketMind - Next-gen PPO Rocket League Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training
  python -m rocketmind.main train
  
  # Launch Streamlit dashboard
  python -m rocketmind.main dashboard
  
  # Deploy to RLBot
  python -m rocketmind.main deploy --checkpoint checkpoints/best_model.pt
  
  # Evaluate model
  python -m rocketmind.main evaluate --checkpoint checkpoints/best_model.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--config', type=str, default='rocketmind/configs/default.yaml',
                              help='Config file path')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/rocketmind',
                              help='Checkpoint directory')
    train_parser.add_argument('--log-dir', type=str, default='logs/rocketmind',
                              help='Log directory')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Resume from checkpoint')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501,
                                  help='Dashboard port')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to RLBot')
    deploy_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Model checkpoint to deploy')
    deploy_parser.add_argument('--config', type=str, default='rocketmind/configs/default.yaml',
                               help='Config file path')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Model checkpoint to evaluate')
    eval_parser.add_argument('--episodes', type=int, default=20,
                             help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        from .train import train
        train(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=args.resume
        )
    
    elif args.command == 'dashboard':
        import subprocess
        print(f"🚀 Launching RocketMind Dashboard on port {args.port}...")
        print(f"   Open http://localhost:{args.port} in your browser")
        subprocess.run([
            'streamlit', 'run',
            'rocketmind/streamlit_app.py',
            '--server.port', str(args.port)
        ])
    
    elif args.command == 'deploy':
        from .rlbot_interface import RLBotLauncher
        
        print("🚀 Deploying RocketMind to RLBot")
        print(f"   Checkpoint: {args.checkpoint}")
        
        launcher = RLBotLauncher(
            checkpoint_path=args.checkpoint,
            config_path=args.config
        )
        
        # Create bot config
        launcher.create_bot_config()
        
        # Launch
        launcher.launch_in_gui()
        
        print("\n✓ Deployment complete!")
        print("  Add the bot through RLBot GUI using bot.cfg")
    
    elif args.command == 'evaluate':
        print(f"📊 Evaluating model: {args.checkpoint}")
        print(f"   Episodes: {args.episodes}")
        print("\n⚠ Evaluation functionality coming soon")
        # Evaluation logic would go here


if __name__ == "__main__":
    main()
