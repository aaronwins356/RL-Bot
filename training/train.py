import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from training.envs.env import RLBotEnv
from training.rewards.rewards import CombinedReward
from training.curriculum.scheduler import CurriculumScheduler
from metrics.evaluator import Evaluator

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_env(config: dict, scenario: str) -> RLBotEnv:
    reward_weights = config['training']['rewards']
    
    return RLBotEnv(
        scenario=scenario,
        reward_fn=CombinedReward(reward_weights),
        obs_builder=config['training']['obs_builder'],
        state_setter=config['training']['state_setter'],
        terminal_condition=TimeoutCondition(config['training']['max_steps']),
        team_size=config['training']['team_size']
    )

def main():
    # Load configuration
    config = load_config('config/training.yaml')
    
    # Set up logging
    log_dir = "training/logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["tensorboard", "csv"])
    
    # Create curriculum scheduler
    curriculum = CurriculumScheduler(config['curriculum'])
    
    # Create environment
    env = create_env(config, curriculum.current_scenario())
    env = Monitor(env)
    
    # Initialize model
    model = PPO(
        "MlpPolicy",
        env,
        **config['model_params'],
        tensorboard_log=log_dir
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['checkpoint_freq'],
        save_path="training/ckpts",
        name_prefix="rlbot"
    )
    
    eval_env = create_env(config, "evaluation")
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        log_path=log_dir,
        best_model_save_path="training/ckpts/best"
    )
    
    # Training loop
    total_timesteps = config['training']['total_timesteps']
    curriculum_update_freq = config['curriculum']['update_freq']
    
    for timestep in range(0, total_timesteps, curriculum_update_freq):
        # Train for curriculum interval
        model.learn(
            total_timesteps=curriculum_update_freq,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=False,
            tb_log_name=curriculum.current_scenario()
        )
        
        # Update curriculum
        if curriculum.should_progress(model):
            env = create_env(config, curriculum.next_scenario())
            model.set_env(env)
            
    # Final evaluation
    evaluator = Evaluator(config['evaluation'])
    evaluator.evaluate_model(model)
    evaluator.generate_report()

if __name__ == "__main__":
    main()