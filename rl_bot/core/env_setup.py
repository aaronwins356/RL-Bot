"""
Rocket League Gym environment setup using rlgym.rocket_league 2.0+.
Creates and configures the RL training environment.
"""

import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces

# New RLGym 2.0+ API imports
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league import common_values

# Keep these for compatibility with custom reward functions and observers
try:
    from rlgym.rocket_league.api import GameState, Car
    PlayerData = Car  # In RLGym 2.0, Car is used instead of PlayerData
except ImportError:
    # Fallback for different API versions
    from rlgym.api import GameState
    class PlayerData:
        pass


# Compatibility base classes for legacy code
class ObsBuilder:
    """Base class for custom observation builders - RLGym 2.0 compatible."""
    def reset(self, initial_state):
        pass
    
    def build_obs(self, player, state, previous_action):
        raise NotImplementedError


class ActionParser:
    """Base class for custom action parsers - RLGym 2.0 compatible."""
    def get_action_space(self):
        raise NotImplementedError
    
    def parse_actions(self, actions, state):
        raise NotImplementedError


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
) -> RLGym:
    """
    Create a Rocket League Gym environment with specified configuration using RLGym 2.0+ API.
    
    Args:
        config: Configuration dictionary
        reward_fn: Reward function (if None, creates default)
        team_size: Number of players per team
        tick_skip: Physics ticks to skip per action (action repeat)
        timeout_seconds: Episode timeout in seconds
        spawn_opponents: Whether to spawn opponent bots
        
    Returns:
        Configured RL environment (RLGym 2.0)
    """
    env_config = config.get('environment', {})
    
    # Team sizes
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    
    # Action parser with repeat
    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    
    # Terminal and truncation conditions
    termination_condition = GoalCondition()
    no_touch_timeout = env_config.get('no_touch_timeout_seconds', 30)
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout),
        TimeoutCondition(timeout_seconds=timeout_seconds)
    )
    
    # Create reward function if not provided
    if reward_fn is None:
        # Simple default reward
        reward_fn = CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 0.1)
        )
    
    # Observation builder
    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )
    
    # State mutators (for resetting environment state)
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator()
    )
    
    # Create environment using new RLGym 2.0 API
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine()
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
            obs_dict = env.reset()
            # RLGym 2.0 returns dict of observations {agent_id: obs}
            if isinstance(obs_dict, dict):
                # Get first agent's observation
                obs = list(obs_dict.values())[0]
            else:
                obs = obs_dict
            observations.append(obs)
        return np.array(observations, dtype=np.float32), {}
    
    def step(self, actions):
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of actions for each environment
            
        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            # RLGym 2.0 expects dict of actions {agent_id: action}
            agent_ids = list(env.agents)
            action_dict = {agent_ids[0]: actions[i]}  # Single agent for now
            
            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(action_dict)
            
            # Extract first agent's data
            obs = list(obs_dict.values())[0]
            reward = list(reward_dict.values())[0]
            terminated = list(terminated_dict.values())[0]
            truncated = list(truncated_dict.values())[0]
            
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append({})
        
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
