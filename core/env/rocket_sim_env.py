"""RocketSim-based Rocket League environment for training.

This module provides a gym-compatible environment wrapper for RocketSim,
with support for aerial training, boost management, and reward shaping.
"""
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import yaml
import gymnasium as gym
from gymnasium import spaces

from core.features.encoder import ObservationEncoder, RawObservation

logger = logging.getLogger(__name__)


def safe_reset(env: gym.Env, max_retries: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Safely reset environment with automatic retry and error handling.
    
    Args:
        env: Gym environment to reset
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (observation, info)
    """
    for attempt in range(max_retries):
        try:
            result = env.reset()
            
            # Handle both old (obs) and new (obs, info) formats
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs = result
                info = {}
            
            # Check for NaN observations
            if isinstance(obs, np.ndarray) and np.any(np.isnan(obs)):
                logger.warning(f"NaN observation detected on reset attempt {attempt + 1}, retrying...")
                continue
                
            return obs, info
            
        except Exception as e:
            logger.error(f"Reset failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, return zero observation
                logger.error("All reset attempts failed, returning zero observation")
                return np.zeros(getattr(env, 'OBS_SIZE', 180), dtype=np.float32), {}
    
    # Should not reach here, but return safe default
    return np.zeros(getattr(env, 'OBS_SIZE', 180), dtype=np.float32), {}


def safe_step(env: gym.Env, action: np.ndarray, max_retries: int = 2) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    """Safely execute environment step with automatic recovery and API adaptation.
    
    This function handles both old and new Gym API formats and recovers from errors.
    
    Args:
        env: Gym environment
        action: Action to execute
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (obs, reward, terminated, truncated, info)
    """
    for attempt in range(max_retries):
        try:
            result = env.step(action)
            
            # Handle different return formats
            if len(result) == 4:
                # Old format: (obs, reward, done, info)
                obs, reward, done, info = result
                return obs, reward, done, False, info
            elif len(result) == 5:
                # New format: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                
                # Check for NaN observations
                if isinstance(obs, np.ndarray) and np.any(np.isnan(obs)):
                    logger.warning(f"NaN observation detected on step attempt {attempt + 1}, resetting...")
                    obs, info = safe_reset(env)
                    return obs, 0.0, True, False, info
                
                return obs, reward, terminated, truncated, info
            else:
                raise ValueError(f"Unexpected env.step() format with {len(result)} elements")
                
        except Exception as e:
            logger.error(f"Env step failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, reset environment
                logger.error("All step attempts failed, resetting environment")
                obs, info = safe_reset(env)
                return obs, 0.0, True, False, info
    
    # Should not reach here, but return safe default
    obs, info = safe_reset(env)
    return obs, 0.0, True, False, info


class RocketSimEnv(gym.Env):
    """Gym-compatible environment for Rocket League using RocketSim.
    
    This environment supports:
    - Aerial training scenarios
    - Boost management
    - Reward shaping for various behaviors
    - Configurable game modes (1v1, 2v2, 3v3)
    
    Attributes:
        observation_space: Observation space (Box)
        action_space: Action space (Box or MultiDiscrete)
        encoder: Observation encoder
        reward_config: Reward shaping configuration
    """
    
    # Define observation and action space sizes as class constants
    OBS_SIZE = 180  # From ObservationEncoder
    ACTION_SIZE = 8  # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    
    def __init__(
        self,
        reward_config_path: Optional[Path] = None,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        spawn_opponents: bool = True,
        enable_aerial_training: bool = True,
        random_spawn: bool = True,
        simulation_mode: bool = True,  # True for simulated episodes, False for real RocketSim
        debug_mode: bool = False,  # Enable detailed logging
    ):
        """Initialize RocketSim environment.
        
        Args:
            reward_config_path: Path to reward configuration YAML
            game_mode: Game mode ("1v1", "2v2", "3v3")
            tick_skip: Number of physics ticks per action
            spawn_opponents: Whether to spawn opponent bots
            enable_aerial_training: Enable aerial training scenarios
            random_spawn: Randomize spawn positions
            simulation_mode: Use simplified simulation for training (True) or real RocketSim (False)
            debug_mode: Enable detailed step-by-step logging
        """
        super().__init__()
        
        self.game_mode = game_mode
        self.tick_skip = tick_skip
        self.spawn_opponents = spawn_opponents
        self.enable_aerial_training = enable_aerial_training
        self.random_spawn = random_spawn
        self.simulation_mode = simulation_mode
        self.debug_mode = debug_mode
        
        # Define observation space (continuous values normalized to [-1, 1] or [0, 1])
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_SIZE,),
            dtype=np.float32
        )
        
        # Define action space (continuous controls)
        # Actions: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(self.ACTION_SIZE,),
            dtype=np.float32
        )
        
        # Load reward configuration
        if reward_config_path and reward_config_path.exists():
            with open(reward_config_path, 'r') as f:
                self.reward_config = yaml.safe_load(f)
        else:
            self.reward_config = self._default_reward_config()
        
        # Initialize encoder
        self.encoder = ObservationEncoder(config={
            'normalize': True,
            'include_history': False
        })
        
        # Environment state
        self.episode_length = 0
        self.max_episode_length = 3000  # ~2 minutes at 120Hz
        self.total_reward = 0.0
        self.prev_ball_velocity_toward_goal = 0.0
        self.prev_boost = 100.0
        self.last_touch_time = 0.0
        self.steps_since_last_touch = 0  # Track idle behavior
        self.aerial_attempts = 0
        self.aerial_successes = 0
        
        # Simulated game state (for simulation mode)
        self.sim_ball_position = np.array([0.0, 0.0, 100.0])
        self.sim_ball_velocity = np.array([0.0, 0.0, 0.0])
        self.sim_car_position = np.array([0.0, -2000.0, 20.0])
        self.sim_car_velocity = np.array([0.0, 0.0, 0.0])
        self.sim_boost_amount = 33.0
        
        # Track previous state for reward calculations
        self.prev_dist_to_ball = 1000.0  # Initial distance approximation
        
        # Goal dimensions (Rocket League standard)
        self.GOAL_WIDTH = 1786.0  # Half-width (893 * 2)
        self.GOAL_HEIGHT = 1284.0  # Half-height (642 * 2)
        
        # Stats for reward calculation
        self.stats = {
            'goals_scored': 0,
            'goals_conceded': 0,
            'aerial_touches': 0,
            'demos_given': 0,
            'demos_taken': 0,
            'boost_collected': 0,
            'touches': 0,
        }
        
    def _default_reward_config(self) -> Dict[str, Any]:
        """Get default reward configuration."""
        return {
            'sparse': {
                'goal_scored': 10.0,
                'goal_conceded': -10.0,
                'demo_opponent': 1.0,
                'demoed_self': -1.0,
                'shot_on_goal': 2.0,
                'save': 3.0,
            },
            'dense': {
                'ball_velocity_toward_goal': 0.01,
                'aerial_touch_bonus': 0.5,
                'boost_pickup': 0.1,
                'boost_waste_penalty': -0.01,
                'touch_bonus': 0.1,
                'goal_proximity': 0.05,
                'velocity_toward_ball': 0.02,
                'distance_to_ball_decrease': 0.01,
            },
            'penalties': {
                'own_goal_risk': -0.5,
                'double_commit': -0.2,
                'missed_aerial': -0.3,
                'boost_starve': -0.1,
                'idle_penalty': -0.05,  # Penalty for not touching ball
                'driving_in_circles': -0.03,
            }
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.episode_length = 0
        self.total_reward = 0.0
        self.prev_ball_velocity_toward_goal = 0.0
        self.prev_boost = 33.0  # Start with 33 boost (standard kickoff)
        self.last_touch_time = 0.0
        self.steps_since_last_touch = 0
        self.aerial_attempts = 0
        self.aerial_successes = 0
        self.prev_dist_to_ball = 2000.0  # Reset previous distance
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
        
        # Reset simulated state
        if self.simulation_mode:
            # Random kickoff positions
            if self.random_spawn:
                self.sim_car_position = np.array([
                    np.random.uniform(-500, 500),
                    np.random.uniform(-3000, -2000),
                    20.0
                ])
            else:
                self.sim_car_position = np.array([0.0, -2000.0, 20.0])
            
            self.sim_car_velocity = np.array([0.0, 0.0, 0.0])
            self.sim_ball_position = np.array([0.0, 0.0, 100.0])
            self.sim_ball_velocity = np.array([0.0, 0.0, 0.0])
            self.sim_boost_amount = 33.0
        
        # Initialize game state
        # In a real implementation, this would initialize RocketSim
        # For now, return a dummy observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'episode_length': 0,
            'total_reward': 0.0,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Action to execute (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure action is properly shaped and clamped
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # Clamp action to valid ranges for stability
        action = np.clip(action, self._action_space.low, self._action_space.high)
        
        # Ensure action has correct length (pad with zeros if needed)
        if len(action) < self.ACTION_SIZE:
            action = np.pad(action, (0, self.ACTION_SIZE - len(action)), mode='constant')
        elif len(action) > self.ACTION_SIZE:
            action = action[:self.ACTION_SIZE]
        
        self.episode_length += 1
        self.steps_since_last_touch += 1
        
        # Execute action in simulation (placeholder or real RocketSim)
        if self.simulation_mode:
            self._simulate_step(action)
        else:
            # In real implementation, would apply action to RocketSim
            pass
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, action)
        self.total_reward += reward
        
        # Debug logging
        if self.debug_mode and self.episode_length % 50 == 0:
            logger.debug(
                f"Step {self.episode_length}: "
                f"reward={reward:.3f}, "
                f"total_reward={self.total_reward:.2f}, "
                f"touches={self.stats['touches']}, "
                f"steps_since_touch={self.steps_since_last_touch}, "
                f"boost={self.sim_boost_amount if self.simulation_mode else 0:.1f}"
            )
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.episode_length >= self.max_episode_length
        
        # Log episode completion
        if (terminated or truncated) and self.debug_mode:
            logger.info(
                f"Episode ended: length={self.episode_length}, "
                f"total_reward={self.total_reward:.2f}, "
                f"goals_scored={self.stats['goals_scored']}, "
                f"goals_conceded={self.stats['goals_conceded']}, "
                f"touches={self.stats['touches']}, "
                f"terminated={terminated}, truncated={truncated}"
            )
        
        # Additional info
        info = {
            'episode_length': self.episode_length,
            'total_reward': self.total_reward,
            'stats': self.stats.copy(),
            'aerial_success_rate': self.aerial_successes / max(1, self.aerial_attempts),
            'steps_since_touch': self.steps_since_last_touch,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_step(self, action: np.ndarray):
        """Simulate a simple physics step for training without RocketSim.
        
        Args:
            action: Action array [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        # Extract action components
        throttle = action[0] if len(action) > 0 else 0.0
        steer = action[1] if len(action) > 1 else 0.0
        boost = action[6] if len(action) > 6 else 0.0
        
        # Simple car physics
        dt = 0.008 * self.tick_skip  # time step
        speed = np.linalg.norm(self.sim_car_velocity)
        max_speed = 2300.0  # Rocket League max speed
        
        # Acceleration
        if throttle > 0 and boost > 0.5 and self.sim_boost_amount > 0:
            # Boosting
            accel = 991.667  # boost acceleration
            self.sim_boost_amount = max(0, self.sim_boost_amount - 33.3 * dt)  # boost consumption
        elif throttle > 0:
            # Normal acceleration
            accel = 1600.0 if speed < 1400 else 160.0
        else:
            # Coasting/braking
            accel = -525.0 if throttle < 0 else 0.0
        
        # Update velocity (simplified 2D)
        forward_dir = np.array([np.sin(steer * 0.1), np.cos(steer * 0.1), 0])
        self.sim_car_velocity += forward_dir * accel * dt
        
        # Clamp speed
        speed = np.linalg.norm(self.sim_car_velocity)
        if speed > max_speed:
            self.sim_car_velocity = self.sim_car_velocity / speed * max_speed
        
        # Update position
        self.sim_car_position += self.sim_car_velocity * dt
        
        # Simple ball physics - ball moves toward opponent goal if car is close
        dist_to_ball = np.linalg.norm(self.sim_ball_position - self.sim_car_position)
        
        if dist_to_ball < 200:  # Touch range
            # Ball was touched
            self.stats['touches'] += 1
            self.steps_since_last_touch = 0
            self.last_touch_time = self.episode_length * dt
            
            # Ball moves in car velocity direction (simplified)
            touch_power = min(speed / max_speed, 1.0)
            self.sim_ball_velocity = self.sim_car_velocity * touch_power * 1.5
            
            # Check if aerial touch
            if self.sim_ball_position[2] > 200:
                self.stats['aerial_touches'] += 1
                self.aerial_successes += 1
        
        # Update ball position
        self.sim_ball_position += self.sim_ball_velocity * dt
        
        # Ball gravity and ground bounce
        self.sim_ball_velocity[2] -= 650.0 * dt  # gravity
        if self.sim_ball_position[2] < 100:
            self.sim_ball_position[2] = 100
            self.sim_ball_velocity[2] = abs(self.sim_ball_velocity[2]) * 0.6  # bounce
        
        # Check for goal (simplified)
        goal_y = 5120  # opponent goal Y position
        if abs(self.sim_ball_position[0]) < self.GOAL_WIDTH / 2 and abs(self.sim_ball_position[2]) < self.GOAL_HEIGHT / 2:
            if self.sim_ball_position[1] > goal_y:
                self.stats['goals_scored'] += 1
            elif self.sim_ball_position[1] < -goal_y:
                self.stats['goals_conceded'] += 1
        
        # Boost pad pickup (simplified - random chance)
        if self.sim_boost_amount < 50 and np.random.random() < 0.01:
            self.sim_boost_amount = min(100, self.sim_boost_amount + 12)  # small pad
            self.stats['boost_collected'] += 1
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Encoded observation vector
        """
        # In real implementation, would get state from RocketSim
        # For simulation mode, use simulated state
        if self.simulation_mode:
            car_pos = self.sim_car_position
            car_vel = self.sim_car_velocity
            ball_pos = self.sim_ball_position
            ball_vel = self.sim_ball_velocity
            boost = self.sim_boost_amount
        else:
            # Placeholder for real RocketSim
            car_pos = np.array([0.0, 0.0, 20.0])
            car_vel = np.array([500.0, 0.0, 0.0])
            ball_pos = np.array([0.0, 1000.0, 100.0])
            ball_vel = np.array([0.0, 500.0, 0.0])
            boost = self.prev_boost
        
        raw_obs = RawObservation(
            car_position=car_pos,
            car_velocity=car_vel,
            car_angular_velocity=np.array([0.0, 0.0, 0.0]),
            car_rotation_matrix=np.eye(3),
            car_boost=boost,
            car_on_ground=True,
            car_has_flip=True,
            car_is_demoed=False,
            ball_position=ball_pos,
            ball_velocity=ball_vel,
            ball_angular_velocity=np.array([0.0, 0.0, 0.0]),
            is_kickoff=(self.episode_length == 0),
            game_time=self.episode_length * 0.008 * self.tick_skip,
            score_self=self.stats['goals_scored'],
            score_opponent=self.stats['goals_conceded'],
            game_phase="KICKOFF" if self.episode_length == 0 else "NEUTRAL"
        )
        
        return self.encoder.encode(raw_obs)
    
    def _calculate_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward for current step.
        
        Args:
            obs: Current observation
            action: Action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Get reward config sections
        sparse = self.reward_config.get('sparse', {})
        dense = self.reward_config.get('dense', {})
        penalties = self.reward_config.get('penalties', {})
        
        # Sparse rewards - Goals
        if self.episode_length > 0:  # Skip first step
            if self.stats['goals_scored'] > 0:
                reward += sparse.get('goal_scored', 10.0)
            if self.stats['goals_conceded'] > 0:
                reward += sparse.get('goal_conceded', -10.0)
        
        # Dense reward - Ball touches
        if self.steps_since_last_touch == 0:  # Just touched ball
            reward += dense.get('touch_bonus', 0.1)
            
            # Extra reward for aerial touches
            if self.simulation_mode and self.sim_ball_position[2] > 200:
                reward += dense.get('aerial_touch_bonus', 0.5)
        
        # Dense reward - Moving toward ball
        if self.simulation_mode:
            dist_to_ball = np.linalg.norm(self.sim_ball_position - self.sim_car_position)
            
            # Reward for getting closer to ball (use tracked previous distance)
            if dist_to_ball < self.prev_dist_to_ball:
                reward += dense.get('distance_to_ball_decrease', 0.01)
            
            # Update previous distance for next step
            self.prev_dist_to_ball = dist_to_ball
            
            # Reward for ball moving toward goal
            goal_pos = np.array([0.0, 5120.0, 0.0])  # Opponent goal
            ball_to_goal = goal_pos - self.sim_ball_position
            ball_to_goal_dist = np.linalg.norm(ball_to_goal[:2])
            
            if ball_to_goal_dist > 0:
                ball_vel_toward_goal = np.dot(
                    self.sim_ball_velocity[:2],
                    ball_to_goal[:2] / ball_to_goal_dist
                )
                
                # Reward for increasing ball velocity toward goal
                if ball_vel_toward_goal > self.prev_ball_velocity_toward_goal:
                    delta = ball_vel_toward_goal - self.prev_ball_velocity_toward_goal
                    reward += delta * dense.get('ball_velocity_toward_goal', 0.01)
                
                self.prev_ball_velocity_toward_goal = ball_vel_toward_goal
        
        # Dense reward - Boost management
        if self.simulation_mode:
            boost_used = max(0, self.prev_boost - self.sim_boost_amount)
            
            if boost_used > 1.0:  # Using boost
                # Penalty for wasting boost (using when not needed)
                if self.prev_dist_to_ball > 1000:  # Far from ball
                    reward += penalties.get('boost_waste', -0.01) * boost_used / 10.0
            
            # Reward for collecting boost when low
            if self.sim_boost_amount > self.prev_boost and self.prev_boost < 50:
                reward += dense.get('boost_pickup', 0.1)
        
        # Penalty - Idle/Not touching ball for too long
        if self.steps_since_last_touch > 150:  # ~10 seconds without touch
            reward += penalties.get('idle_penalty', -0.05)
        
        # Penalty - Boost starvation
        if self.simulation_mode and self.sim_boost_amount < 10:
            reward += penalties.get('boost_starve', -0.1)
        
        # Update previous boost for next step
        if self.simulation_mode:
            self.prev_boost = self.sim_boost_amount
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if episode is terminated
        """
        # Terminate on goal scored/conceded (sparse reward event)
        if self.stats['goals_scored'] > 0 or self.stats['goals_conceded'] > 0:
            return True
        
        # Terminate if ball went completely out of bounds (simulation error)
        if self.simulation_mode:
            if abs(self.sim_ball_position[0]) > 8000 or abs(self.sim_ball_position[1]) > 10000:
                return True
        
        # Force termination if no ball touches for extended period (agent stuck/idle)
        if self.steps_since_last_touch > 300:  # ~20 seconds
            return True
        
        return False
    
    def _detect_aerial_opportunity(self, ball_height: float, ball_distance: float) -> bool:
        """Detect if there's an aerial opportunity.
        
        Args:
            ball_height: Height of ball above ground
            ball_distance: Distance to ball
            
        Returns:
            True if aerial opportunity exists
        """
        # Ball must be airborne and reachable
        min_aerial_height = 200.0  # Minimum height to consider aerial
        max_aerial_distance = 2000.0  # Maximum distance to reach
        
        return ball_height > min_aerial_height and ball_distance < max_aerial_distance
    
    def _calculate_boost_efficiency(self, boost_used: float, action_value: float) -> float:
        """Calculate boost usage efficiency.
        
        Args:
            boost_used: Amount of boost consumed
            action_value: Value of action taken (e.g., goal proximity gain)
            
        Returns:
            Efficiency score (0-1)
        """
        if boost_used <= 0:
            return 1.0
        
        # Efficiency is action value per unit boost
        efficiency = action_value / boost_used
        return np.clip(efficiency, 0.0, 1.0)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment.
        
        Args:
            mode: Render mode ('human', 'rgb_array')
            
        Returns:
            Rendered frame if mode='rgb_array', None otherwise
        """
        # In real implementation, would render game state
        pass
    
    def close(self):
        """Close environment and cleanup resources."""
        # Cleanup RocketSim instance
        pass
    
    @property
    def observation_space(self):
        """Get observation space.
        
        Returns:
            Observation space definition
        """
        return self._observation_space
    
    @property
    def action_space(self):
        """Get action space.
        
        Returns:
            Action space definition
        """
        return self._action_space
