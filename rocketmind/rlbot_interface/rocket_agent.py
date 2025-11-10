"""
Rocket Agent - Main RLBot agent implementation for RocketMind.
Integrates PPO model with RLBot Framework for live play.
"""

import numpy as np
from pathlib import Path
from typing import Optional

try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False
    BaseAgent = object  # Fallback

from .rlbot_adapter import RLBotAdapter, load_trained_model
from .state_parser import StateParser
from ..ppo_core import load_config, get_device


class RocketAgent(BaseAgent if RLBOT_AVAILABLE else object):
    """
    RocketMind agent for RLBot Framework.
    Deploys trained PPO model in actual Rocket League matches.
    """
    
    def __init__(
        self,
        name: str,
        team: int,
        index: int,
        checkpoint_path: str = "checkpoints/best_model.pt",
        config_path: str = "rocketmind/configs/default.yaml"
    ):
        """
        Args:
            name: Bot name
            team: Team number (0 = blue, 1 = orange)
            index: Bot index in match
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        if RLBOT_AVAILABLE:
            super().__init__(name, team, index)
        
        self.name = name
        self.team = team
        self.index = index
        
        print(f"[{self.name}] Initializing RocketMind agent...")
        
        # Load configuration
        try:
            self.config = load_config(config_path)
            print(f"[{self.name}] ✓ Loaded config: {config_path}")
        except Exception as e:
            print(f"[{self.name}] ✗ Failed to load config: {e}")
            self.config = {}
        
        # Setup device
        self.device = get_device(self.config.get('device', 'cpu'))  # Use CPU for RLBot
        print(f"[{self.name}] ✓ Device: {self.device}")
        
        # Load trained model
        try:
            self.model = load_trained_model(checkpoint_path, self.config, self.device)
            print(f"[{self.name}] ✓ Model loaded: {checkpoint_path}")
        except Exception as e:
            print(f"[{self.name}] ✗ Failed to load model: {e}")
            self.model = None
        
        # Create state parser
        team_size = self.config.get('environment', {}).get('team_size', 1)
        self.state_parser = StateParser(team_size=team_size)
        print(f"[{self.name}] ✓ State parser initialized")
        
        # Create RLBot adapter
        if self.model is not None:
            # Use action parser from rl_bot environment setup (compatible with new API)
            from rl_bot.core.env_setup import DiscreteActionParser
            action_parser = DiscreteActionParser()
            
            tick_skip = self.config.get('rlbot', {}).get('tick_skip', 8)
            self.adapter = RLBotAdapter(self.model, self.device, action_parser, tick_skip)
            print(f"[{self.name}] ✓ RLBot adapter initialized")
        else:
            self.adapter = None
        
        # State tracking
        self.prev_packet = None
        self.frame_count = 0
        
        print(f"[{self.name}] ✓ Initialization complete!")
    
    def get_output(self, packet: 'GameTickPacket') -> 'SimpleControllerState':
        """
        Main method called by RLBot each frame.
        
        Args:
            packet: Current game state
            
        Returns:
            controller: Controller inputs for this frame
        """
        if not RLBOT_AVAILABLE:
            return {}
        
        self.frame_count += 1
        
        # Skip frames according to tick_skip
        tick_skip = self.config.get('rlbot', {}).get('tick_skip', 8)
        if self.frame_count % tick_skip != 0:
            # Return previous action or no-op
            return SimpleControllerState()
        
        # Check if model is loaded
        if self.adapter is None:
            print(f"[{self.name}] Model not loaded, returning no-op")
            return SimpleControllerState()
        
        try:
            # Parse game state to observation
            observation = self.state_parser.parse_packet(packet, self.index)
            
            # Get action from model
            action_idx, value = self.adapter.predict_action(observation, deterministic=True)
            
            # Convert action to controller
            controller_dict = self.adapter.action_to_controller(action_idx)
            controller = self.adapter.create_controller_state(controller_dict)
            
            # Update previous packet
            self.prev_packet = packet
            
            return controller
            
        except Exception as e:
            print(f"[{self.name}] Error in get_output: {e}")
            import traceback
            traceback.print_exc()
            return SimpleControllerState()
    
    def initialize_agent(self):
        """Called once when the bot starts up."""
        if RLBOT_AVAILABLE:
            print(f"[{self.name}] Agent initialized and ready!")
    
    def retire(self):
        """Called when the bot is shutting down."""
        if RLBOT_AVAILABLE:
            print(f"[{self.name}] Agent retiring...")


# Factory function for easier instantiation
def create_rocket_agent(
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "rocketmind/configs/default.yaml"
):
    """
    Create RocketAgent with default paths.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        config_path: Path to config file
        
    Returns:
        agent_class: RocketAgent class configured with paths
    """
    class ConfiguredRocketAgent(RocketAgent):
        def __init__(self, name, team, index):
            super().__init__(name, team, index, checkpoint_path, config_path)
    
    return ConfiguredRocketAgent


# For direct RLBot compatibility
if __name__ == "__main__":
    """
    Entry point for RLBot when launching the bot.
    """
    print("RocketMind - PPO Rocket League Bot")
    print("Compatible with RLBot Framework")
    print()
    print("To use this bot:")
    print("1. Train a model: python rocketmind/train.py")
    print("2. Create bot.cfg pointing to this file")
    print("3. Add bot in RLBot GUI")
    print()
