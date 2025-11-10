"""
RLBot Adapter - Interface between RLBot Framework and PPO agent.
Handles RLBot communication, state parsing, and action conversion.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False
    print("⚠ RLBot not available. Install with: pip install rlbot")

from ..ppo_core import load_checkpoint, get_device, ActorCritic
import torch


class RLBotAdapter:
    """
    Adapter to interface PPO agent with RLBot Framework.
    Converts RLBot game state to observations and actions to controller inputs.
    """
    
    def __init__(
        self,
        model: ActorCritic,
        device: torch.device,
        action_parser: Any,
        tick_skip: int = 8
    ):
        """
        Args:
            model: Trained ActorCritic model
            device: PyTorch device
            action_parser: Action parser (from rlgym)
            tick_skip: Ticks per action (match training config)
        """
        self.model = model
        self.device = device
        self.action_parser = action_parser
        self.tick_skip = tick_skip
        
        self.model.eval()
        
        print(f"RLBot Adapter initialized:")
        print(f"  Device: {device}")
        print(f"  Tick skip: {tick_skip}")
    
    def predict_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, float]:
        """
        Predict action from observation.
        
        Args:
            observation: Game state observation
            deterministic: Use deterministic policy
            
        Returns:
            action: Discrete action index
            value: Estimated state value
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            action, _, _, value = self.model.get_action_and_value(
                obs_tensor,
                deterministic=deterministic
            )
            return action.item(), value.item()
    
    def action_to_controller(self, action_idx: int) -> Dict[str, float]:
        """
        Convert discrete action index to controller inputs.
        
        Args:
            action_idx: Discrete action index
            
        Returns:
            controller_dict: Dictionary with controller inputs
                - throttle: [-1, 1]
                - steer: [-1, 1]
                - pitch: [-1, 1]
                - yaw: [-1, 1]
                - roll: [-1, 1]
                - jump: bool
                - boost: bool
                - handbrake: bool
        """
        # Use action parser to convert index to actions
        actions = self.action_parser.parse_actions(np.array([action_idx]), None)
        action = actions[0]
        
        return {
            'throttle': float(action[0]),
            'steer': float(action[1]),
            'pitch': float(action[2]),
            'yaw': float(action[3]),
            'roll': float(action[4]),
            'jump': bool(action[5]),
            'boost': bool(action[6]),
            'handbrake': bool(action[7])
        }
    
    @staticmethod
    def create_controller_state(controller_dict: Dict[str, float]):
        """
        Create RLBot SimpleControllerState from controller dictionary.
        
        Args:
            controller_dict: Controller inputs
            
        Returns:
            SimpleControllerState for RLBot
        """
        if not RLBOT_AVAILABLE:
            return controller_dict  # Return dict if RLBot not available
        
        controller = SimpleControllerState()
        controller.throttle = controller_dict['throttle']
        controller.steer = controller_dict['steer']
        controller.pitch = controller_dict['pitch']
        controller.yaw = controller_dict['yaw']
        controller.roll = controller_dict['roll']
        controller.jump = controller_dict['jump']
        controller.boost = controller_dict['boost']
        controller.handbrake = controller_dict['handbrake']
        
        return controller


class RLBotLauncher:
    """
    Launcher for starting bot in RLBot GUI.
    Provides methods to configure and launch the bot.
    """
    
    def __init__(
        self,
        bot_name: str = "RocketMind",
        bot_description: str = "Next-gen PPO Rocket League bot",
        checkpoint_path: str = "checkpoints/best_model.pt",
        config_path: str = "rocketmind/configs/default.yaml"
    ):
        """
        Args:
            bot_name: Bot name for RLBot
            bot_description: Bot description
            checkpoint_path: Path to trained checkpoint
            config_path: Path to config file
        """
        self.bot_name = bot_name
        self.bot_description = bot_description
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
    
    def create_bot_config(self, output_path: str = "bot.cfg"):
        """
        Create RLBot configuration file.
        
        Args:
            output_path: Where to save bot.cfg
        """
        config_content = f"""[Locations]
python_file = rocketmind/rlbot_interface/rocket_agent.py

[Details]
name = {self.bot_name}
description = {self.bot_description}
fun_fact = Trained with PPO reinforcement learning
github = https://github.com/aaronwins356/RL-Bot
language = python

[Loadout]
# Optional: Customize bot appearance
team_color_id = 0
custom_color_id = 0
car_id = 23  # Octane
decal_id = 0
wheels_id = 0
boost_id = 0
antenna_id = 0
hat_id = 0
paint_finish_id = 0
custom_finish_id = 0
engine_audio_id = 0
trails_id = 0
goal_explosion_id = 0

[Bot Parameters]
checkpoint_path = {self.checkpoint_path}
config_path = {self.config_path}
"""
        
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Bot configuration saved to: {output_path}")
        return output_path
    
    def launch_in_gui(self):
        """
        Launch bot in RLBot GUI.
        This requires RLBot to be installed and configured.
        """
        if not RLBOT_AVAILABLE:
            print("✗ RLBot not installed. Install with: pip install rlbot")
            return False
        
        print(f"Launching {self.bot_name} in RLBot GUI...")
        print("Please add the bot through the RLBot GUI interface.")
        print(f"Bot config: bot.cfg")
        
        # In a real implementation, this would use RLBot's Python API
        # to programmatically add the bot to a match
        return True


def load_trained_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> ActorCritic:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        model: Loaded ActorCritic model
    """
    from ..ppo_core import create_actor_critic
    
    # Create model
    # Note: obs_dim and action_dim need to be determined from environment
    # For now, use placeholder values - these should match training
    obs_dim = 107  # Default for simple observation
    action_dim = 90  # Default for rlgym discrete action space
    
    model = create_actor_critic(obs_dim, action_dim, config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Timesteps trained: {checkpoint.get('total_timesteps', 'unknown')}")
    
    return model
