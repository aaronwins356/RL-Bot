"""
Rocket League Gym environment setup using rlgym.
Creates and configures the RL training environment.
"""

import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces

from rlgym_sim.envs import RLGymSimEnv
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.action_parsers import ActionParser
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    TimeoutCondition, GoalScoredCondition
)
from rlgym_sim.utils.reward_functions import CombinedReward

from rlgym_sim.utils.gamestates import GameState, PlayerData


class SimpleObsBuilder(ObsBuilder):
    """
    Simple observation builder that creates a flat observation vector.
    Includes player state, ball state, and relative information.
    """
    
    def __init__(self):
        super().__init__()
        # Observation dimensions:
        # Player: position(3) + velocity(3) + rotation(9) + angular_vel(3) + boost(1) + on_ground(1) = 20
        # Ball: position(3) + velocity(3) + angular_vel(3) = 9
        # Relative: ball_pos_rel(3) + ball_vel_rel(3) = 6
        # Total = 35 dimensions
        self.obs_dim = 35
    
    def reset(self, initial_state: GameState):
        """Reset observation builder."""
        pass
    
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Build observation for a single player.
        
        Args:
            player: Current player data
            state: Current game state
            previous_action: Previous action taken
            
        Returns:
            Flat observation array
        """
        obs = []
        
        # Player car data (normalized to roughly -1 to 1 range)
        car = player.car_data
        obs.extend(car.position / 4096)  # Field is ~8192 x 10240, normalize to [-1, 1]
        obs.extend(car.linear_velocity / 2300)  # Max car speed ~2300
        obs.extend(car.rotation_mtx().flatten())  # 3x3 rotation matrix flattened
        obs.extend(car.angular_velocity / 5.5)  # Max angular velocity ~5.5
        obs.append(player.boost_amount / 100)  # Boost normalized to [0, 1]
        obs.append(float(player.on_ground))  # Binary flag
        
        # Ball data (normalized)
        ball = state.ball
        obs.extend(ball.position / 4096)
        obs.extend(ball.linear_velocity / 6000)  # Max ball speed ~6000
        obs.extend(ball.angular_velocity / 6)
        
        # Relative ball information
        ball_rel_pos = (ball.position - car.position) / 4096
        ball_rel_vel = (ball.linear_velocity - car.linear_velocity) / 2300
        obs.extend(ball_rel_pos)
        obs.extend(ball_rel_vel)
        
        return np.asarray(obs, dtype=np.float32)


class DiscreteActionParser(ActionParser):
    """
    Discrete action parser with a simplified action space.
    Maps discrete actions to continuous controller inputs.
    """
    
    def __init__(self):
        super().__init__()
        # Define discrete actions (90 total):
        # 5 throttle values x 3 steer values x 3 pitch values x 2 boost x 1 jump
        # Simplified for readability: we'll use a smaller action space
        self.actions = self._create_action_set()
        self.n_actions = len(self.actions)
    
    def _create_action_set(self):
        """Create the set of discrete actions."""
        actions = []
        
        # Throttle: -1, 0, 1
        throttles = [-1, 0, 1]
        # Steer: -1, 0, 1  
        steers = [-1, 0, 1]
        # Pitch: -1, 0, 1
        pitches = [-1, 0, 1]
        # Yaw: 0 (simplified)
        yaw = 0
        # Roll: 0 (simplified)
        roll = 0
        # Jump: 0, 1
        jumps = [0, 1]
        # Boost: 0, 1
        boosts = [0, 1]
        # Handbrake: 0 (simplified)
        handbrake = 0
        
        for throttle in throttles:
            for steer in steers:
                for pitch in pitches:
                    for jump in jumps:
                        for boost in boosts:
                            action = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
                            actions.append(action)
        
        return actions
    
    def get_action_space(self) -> spaces.Space:
        """Return the action space."""
        return spaces.Discrete(self.n_actions)
    
    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        """
        Parse discrete actions to continuous controls.
        
        Args:
            actions: Array of discrete action indices
            state: Current game state
            
        Returns:
            Array of continuous actions
        """
        parsed_actions = []
        for action_idx in actions:
            parsed_actions.append(self.actions[int(action_idx)])
        return np.asarray(parsed_actions, dtype=np.float32)


def create_rlgym_env(
    config: Dict[str, Any],
    reward_fn: Optional[Any] = None,
    team_size: int = 1,
    tick_skip: int = 8,
    timeout_seconds: int = 300,
    spawn_opponents: bool = True
) -> gym.Env:
    """
    Create a Rocket League Gym environment with specified configuration.
    
    Args:
        config: Configuration dictionary
        reward_fn: Reward function (if None, creates default)
        team_size: Number of players per team
        tick_skip: Physics ticks to skip per action
        timeout_seconds: Episode timeout in seconds
        spawn_opponents: Whether to spawn opponent bots
        
    Returns:
        Configured RL environment
    """
    env_config = config.get('environment', {})
    
    # Create observation builder based on config
    obs_builder_type = env_config.get('obs_builder', 'simple')
    include_predictions = env_config.get('include_predictions', True)
    
    if obs_builder_type == 'team_aware':
        from rl_bot.core.advanced_obs import TeamAwareObsBuilder
        max_team_size = env_config.get('max_team_size', 3)
        obs_builder = TeamAwareObsBuilder(
            max_team_size=max_team_size,
            include_predictions=include_predictions,
            num_predictions=5
        )
    elif obs_builder_type == 'compact':
        from rl_bot.core.advanced_obs import CompactObsBuilder
        obs_builder = CompactObsBuilder(include_predictions=include_predictions)
    else:
        # Default simple obs builder
        obs_builder = SimpleObsBuilder()
    
    # Create action parser
    action_parser = DiscreteActionParser()
    
    # Create reward function if not provided
    if reward_fn is None:
        from rl_bot.core.reward_functions import create_reward_function
        reward_fn = create_reward_function(config)
    
    # Create terminal conditions
    terminal_conditions = [
        TimeoutCondition(timeout_seconds * 120 // tick_skip),  # 120 ticks per second
        GoalScoredCondition(),
    ]
    
    # Create state setter (spawning configuration)
    state_setter = DefaultState()
    
    # Create environment
    env = RLGymSimEnv(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        terminal_conditions=terminal_conditions,
        state_setter=state_setter
    )
    
    return env


class VectorizedEnv:
    """
    Simple vectorized environment wrapper for parallel training.
    """
    
    def __init__(self, env_fns, obs_dim, action_dim):
        """
        Args:
            env_fns: List of functions that create environments
            obs_dim: Observation space dimension
            action_dim: Action space dimension (for discrete, this is num_actions)
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)
    
    def reset(self):
        """Reset all environments."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            # Handle both single and multi-agent obs
            if isinstance(obs, (list, tuple)):
                obs = obs[0]  # Take first agent
            observations.append(obs)
        return np.array(observations, dtype=np.float32), {}
    
    def step(self, actions):
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of actions for each environment
            
        Returns:
            observations, rewards, dones, truncated, infos
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            
            # Handle multi-agent environments
            if isinstance(obs, (list, tuple)):
                obs = obs[0]
            if isinstance(reward, (list, tuple)):
                reward = reward[0]
            if isinstance(terminated, (list, tuple)):
                terminated = terminated[0]
            if isinstance(truncated, (list, tuple)):
                truncated = truncated[0]
            
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info if isinstance(info, dict) else {})
        
        return (
            np.array(observations, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            infos
        )
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_vec_env(config: Dict[str, Any], num_envs: int = 4) -> VectorizedEnv:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        config: Configuration dictionary
        num_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    env_config = config.get('environment', {})
    
    # Create environment factory
    def make_env():
        return create_rlgym_env(
            config=config,
            team_size=env_config.get('team_size', 1),
            tick_skip=env_config.get('tick_skip', 8),
            timeout_seconds=env_config.get('timeout_seconds', 300),
            spawn_opponents=env_config.get('spawn_opponents', True)
        )
    
    # Create vectorized environment
    env_fns = [make_env for _ in range(num_envs)]
    
    # Get observation and action dimensions from a temporary environment
    temp_env = make_env()
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()
    
    vec_env = VectorizedEnv(env_fns, obs_dim, action_dim)
    
    return vec_env
