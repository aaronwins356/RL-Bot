#!/usr/bin/env python3
"""
Example script showing how to use the RL-Bot system.
This demonstrates the basic workflow for training and evaluation.
"""

import sys
from pathlib import Path

# Example 1: Basic Training
def example_basic_training():
    """Example: Train a bot with default settings."""
    print("=" * 60)
    print("Example 1: Basic Training")
    print("=" * 60)
    print()
    print("To train with default settings:")
    print("  python main.py")
    print()
    print("This will:")
    print("  - Load config.yaml")
    print("  - Create 4 parallel environments")
    print("  - Train for 10M timesteps")
    print("  - Save checkpoints to checkpoints/")
    print("  - Log to logs/")
    print()


# Example 2: Custom Training
def example_custom_training():
    """Example: Train with custom settings."""
    print("=" * 60)
    print("Example 2: Custom Training")
    print("=" * 60)
    print()
    print("To train with custom settings:")
    print("  python main.py --timesteps 5000000 --device cuda --num-envs 8")
    print()
    print("This will:")
    print("  - Train for 5M timesteps")
    print("  - Use GPU acceleration")
    print("  - Use 8 parallel environments")
    print()


# Example 3: Programmatic Training
def example_programmatic_training():
    """Example: Train programmatically."""
    print("=" * 60)
    print("Example 3: Programmatic Training")
    print("=" * 60)
    print()
    
    code = """
from rl_bot.core.utils import load_config, get_device, setup_logging, set_seed
from rl_bot.core.env_setup import make_vec_env
from rl_bot.train import train

# Setup
config = load_config('config.yaml')
set_seed(42)
device = get_device('auto')
logger = setup_logging('logs')

# Create environment
env = make_vec_env(config, num_envs=4)

# Train
model = train(
    env=env,
    config=config,
    total_timesteps=1_000_000,
    device=device,
    logger=logger
)

print("Training complete!")
env.close()
"""
    
    print("Save this to train_custom.py:")
    print(code)
    print()


# Example 4: Evaluation
def example_evaluation():
    """Example: Evaluate a trained model."""
    print("=" * 60)
    print("Example 4: Evaluation")
    print("=" * 60)
    print()
    
    code = """
from rl_bot.core.utils import load_config
from rl_bot.core.env_setup import make_vec_env
from rl_bot.eval import evaluate_checkpoint

# Load config
config = load_config('config.yaml')

# Create environment for evaluation
env = make_vec_env(config, num_envs=1)

# Evaluate
metrics = evaluate_checkpoint(
    'checkpoints/best_model.pt',
    env,
    config,
    num_episodes=20,
    plot_elo=True,
    save_dir='eval_results'
)

print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Elo Rating: {metrics['elo_rating']:.0f}")

env.close()
"""
    
    print("Save this to evaluate.py and run:")
    print("  python evaluate.py")
    print()
    print("Code:")
    print(code)
    print()


# Example 5: Custom Rewards
def example_custom_rewards():
    """Example: Add custom reward function."""
    print("=" * 60)
    print("Example 5: Custom Rewards")
    print("=" * 60)
    print()
    
    code = """
# Add to rl_bot/core/reward_functions.py

class DribbleReward(RewardFunction):
    '''Reward for dribbling the ball.'''
    
    def __init__(self, dribble_reward: float = 0.5):
        super().__init__()
        self.dribble_reward = dribble_reward
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        
        # Check if player is close to ball and ball is on car
        distance = np.linalg.norm(ball_pos - player_pos)
        ball_height = ball_pos[2]
        
        # Reward if ball is on car (close distance, moderate height)
        if distance < 150 and 100 < ball_height < 200:
            return self.dribble_reward
        
        return 0.0

# Add to create_reward_function:
components = [
    # ... existing components
    ('dribble', DribbleReward(
        dribble_reward=config.get('rewards', {}).get('dribble', 0.5)
    ))
]
"""
    
    print("Add custom reward to reward_functions.py:")
    print(code)
    print()
    print("Then add to config.yaml:")
    print("  rewards:")
    print("    dribble: 0.5")
    print()


# Example 6: Resume Training
def example_resume_training():
    """Example: Resume from checkpoint."""
    print("=" * 60)
    print("Example 6: Resume Training")
    print("=" * 60)
    print()
    print("To resume training from a checkpoint:")
    print("  python main.py --resume checkpoints/checkpoint_5000000.pt")
    print()
    print("This will:")
    print("  - Load the model weights")
    print("  - Load the optimizer state")
    print("  - Continue from step 5,000,000")
    print()


# Example 7: TensorBoard Monitoring
def example_tensorboard():
    """Example: Monitor training with TensorBoard."""
    print("=" * 60)
    print("Example 7: TensorBoard Monitoring")
    print("=" * 60)
    print()
    print("While training is running, open a new terminal:")
    print("  tensorboard --logdir logs")
    print()
    print("Then open your browser to:")
    print("  http://localhost:6006")
    print()
    print("You'll see graphs for:")
    print("  - Mean reward over time")
    print("  - Policy loss")
    print("  - Value loss")
    print("  - Learning rate")
    print("  - Entropy")
    print("  - KL divergence")
    print()


# Example 8: Using Behaviors
def example_behaviors():
    """Example: Configure and use behavior system."""
    print("=" * 60)
    print("Example 8: Using Behaviors")
    print("=" * 60)
    print()
    print("The bot can use hardcoded behaviors for specific situations.")
    print()
    print("Configure in config.yaml:")
    print("""
behaviors:
  enabled: true
  kickoff_enabled: true      # Fast kickoff routine
  recovery_enabled: true     # Aerial recovery
  boost_management_enabled: false  # Experimental
""")
    print()
    print("Behaviors automatically override the learned policy when:")
    print("  - Kickoff is detected → Rush to ball with boost")
    print("  - Car is tumbling in air → Reorient wheels down")
    print("  - Boost is low and safe → Navigate to boost pads")
    print()
    print("Disable all behaviors to use pure learned policy:")
    print("  behaviors.enabled: false")
    print()


# Example 9: Team Play (2v2, 3v3)
def example_team_play():
    """Example: Train for team play."""
    print("=" * 60)
    print("Example 9: Team Play Configuration")
    print("=" * 60)
    print()
    print("Train for 2v2 or 3v3 with team-aware observations.")
    print()
    print("In config.yaml:")
    print("""
environment:
  team_size: 2              # 2v2 (or 3 for 3v3)
  obs_builder: "team_aware" # Include teammate/opponent info
  include_predictions: true # Ball trajectory predictions
  max_team_size: 3          # Max players per team
  
rewards:
  positioning_weight: 0.1   # Reward good positioning
  rotation_weight: 0.05     # Reward proper rotation
""")
    print()
    print("Team-aware observations include:")
    print("  - Own player state")
    print("  - Ball state and predictions")
    print("  - Teammate positions and velocities")
    print("  - Opponent positions and velocities")
    print()
    print("This helps the bot learn:")
    print("  - Not to double-commit")
    print("  - When to rotate back")
    print("  - Proper defensive coverage")
    print()


# Example 10: Ball Prediction
def example_ball_prediction():
    """Example: Use ball prediction in custom code."""
    print("=" * 60)
    print("Example 10: Ball Prediction")
    print("=" * 60)
    print()
    print("Ball prediction can be used in custom rewards or behaviors.")
    print()
    print("Example code:")
    print("""
from rl_bot.core.ball_prediction import SimpleBallPredictor
import numpy as np

# Create predictor
predictor = SimpleBallPredictor()

# Current ball state
ball_pos = np.array([0, 0, 500])
ball_vel = np.array([1000, 500, 200])
ball_ang = np.array([0, 0, 0])

# Predict 1 second ahead (120 ticks)
predictions = predictor.predict(
    ball_pos, ball_vel, ball_ang,
    num_steps=120
)

# Check where ball will be in 0.5 seconds
future_pos = predictions[60].position
print(f"Ball will be at: {future_pos}")

# Find when ball lands
landing = predictor.get_landing_prediction(
    ball_pos, ball_vel, ball_ang
)
if landing:
    print(f"Ball lands at: {landing.position}")
""")
    print()
    print("Use predictions for:")
    print("  - Planning aerial intercepts")
    print("  - Defensive positioning")
    print("  - Shot setup")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("RL-Bot Usage Examples".center(60))
    print("*" * 60)
    print("\n")
    
    examples = [
        example_basic_training,
        example_custom_training,
        example_programmatic_training,
        example_evaluation,
        example_custom_rewards,
        example_resume_training,
        example_tensorboard,
        example_behaviors,
        example_team_play,
        example_ball_prediction
    ]
    
    for example in examples:
        example()
        print()
    
    print("*" * 60)
    print("For more details, see README.md and SETUP_GUIDE.md")
    print("*" * 60)
    print()


if __name__ == "__main__":
    main()
