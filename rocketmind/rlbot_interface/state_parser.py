"""
State Parser - Convert RLBot game state to observation vectors.
Handles player data, ball physics, and game information parsing.
"""

import numpy as np
from typing import Dict, Any, Optional, List

try:
    from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False


class StateParser:
    """
    Parse RLBot GameTickPacket into observation vectors.
    Compatible with rlgym observation builders.
    """
    
    def __init__(self, team_size: int = 1, include_predictions: bool = True):
        """
        Args:
            team_size: Number of players per team (1v1, 2v2, or 3v3)
            include_predictions: Include ball prediction in observations
        """
        self.team_size = team_size
        self.include_predictions = include_predictions
    
    def parse_packet(
        self,
        packet,
        player_index: int
    ) -> np.ndarray:
        """
        Parse game packet to observation vector.
        
        Args:
            packet: RLBot GameTickPacket
            player_index: Index of the player we're controlling
            
        Returns:
            observation: Numpy array observation
        """
        if not RLBOT_AVAILABLE:
            # Return dummy observation if RLBot not available
            return np.zeros(107, dtype=np.float32)
        
        # Extract player data
        player = packet.game_cars[player_index]
        
        # Player state (position, velocity, rotation)
        player_obs = self._parse_player(player)
        
        # Ball state
        ball_obs = self._parse_ball(packet.game_ball)
        
        # Boost pads (simplified)
        boost_obs = self._parse_boosts(packet)
        
        # Combine observations
        obs = np.concatenate([player_obs, ball_obs, boost_obs])
        
        return obs.astype(np.float32)
    
    def _parse_player(self, player) -> np.ndarray:
        """Extract player state features."""
        # Position (3)
        pos = np.array([
            player.physics.location.x,
            player.physics.location.y,
            player.physics.location.z
        ])
        
        # Velocity (3)
        vel = np.array([
            player.physics.velocity.x,
            player.physics.velocity.y,
            player.physics.velocity.z
        ])
        
        # Rotation (pitch, yaw, roll) (3)
        rot = np.array([
            player.physics.rotation.pitch,
            player.physics.rotation.yaw,
            player.physics.rotation.roll
        ])
        
        # Angular velocity (3)
        ang_vel = np.array([
            player.physics.angular_velocity.x,
            player.physics.angular_velocity.y,
            player.physics.angular_velocity.z
        ])
        
        # Boost amount (1)
        boost = np.array([player.boost / 100.0])
        
        # On ground, has flip (2)
        on_ground = np.array([float(player.has_wheel_contact)])
        has_flip = np.array([float(player.jumped)])  # Simplified
        
        # Normalize positions and velocities
        pos = pos / 4096.0  # Field scale
        vel = vel / 2300.0  # Max velocity
        ang_vel = ang_vel / 5.5  # Max angular velocity
        
        return np.concatenate([pos, vel, rot, ang_vel, boost, on_ground, has_flip])
    
    def _parse_ball(self, ball) -> np.ndarray:
        """Extract ball state features."""
        # Position (3)
        pos = np.array([
            ball.physics.location.x,
            ball.physics.location.y,
            ball.physics.location.z
        ])
        
        # Velocity (3)
        vel = np.array([
            ball.physics.velocity.x,
            ball.physics.velocity.y,
            ball.physics.velocity.z
        ])
        
        # Angular velocity (3)
        ang_vel = np.array([
            ball.physics.angular_velocity.x,
            ball.physics.angular_velocity.y,
            ball.physics.angular_velocity.z
        ])
        
        # Normalize
        pos = pos / 4096.0
        vel = vel / 6000.0  # Max ball velocity
        ang_vel = ang_vel / 6.0
        
        return np.concatenate([pos, vel, ang_vel])
    
    def _parse_boosts(self, packet) -> np.ndarray:
        """Extract boost pad states (simplified)."""
        # In a full implementation, this would parse all boost pad states
        # For now, return a simplified version
        
        # Large boost pads (6 pads, boolean active state)
        large_boosts = np.zeros(6, dtype=np.float32)
        
        # Small boost pads (aggregated or sampled)
        small_boosts = np.zeros(4, dtype=np.float32)
        
        return np.concatenate([large_boosts, small_boosts])
    
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        player_dim = 15  # Player state
        ball_dim = 9     # Ball state
        boost_dim = 10   # Boost pads
        
        base_dim = player_dim + ball_dim + boost_dim
        
        # Add teammates and opponents for team play
        if self.team_size > 1:
            # Simplified: add space for teammates and opponents
            base_dim += (self.team_size - 1) * player_dim  # Teammates
            base_dim += self.team_size * player_dim        # Opponents
        
        # Add ball predictions if enabled
        if self.include_predictions:
            base_dim += 18  # 6 future positions (3D each)
        
        return base_dim


class RewardCalculator:
    """
    Calculate rewards from RLBot game state.
    Can be used for online learning or evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration with reward weights
        """
        self.config = config
        self.reward_weights = config.get('rewards', {})
        
        # Track previous state for reward calculation
        self.prev_ball_vel_to_goal = 0.0
        self.prev_player_touched_ball = False
    
    def calculate_reward(
        self,
        packet,
        player_index: int,
        prev_packet
    ) -> float:
        """
        Calculate reward from state transition.
        
        Args:
            packet: Current game packet
            player_index: Player index
            prev_packet: Previous game packet
            
        Returns:
            reward: Scalar reward
        """
        reward = 0.0
        
        if not RLBOT_AVAILABLE:
            return reward
        
        # Goal scored/conceded
        current_score = packet.teams[0].score + packet.teams[1].score
        prev_score = prev_packet.teams[0].score + prev_packet.teams[1].score if prev_packet else 0
        
        if current_score > prev_score:
            # Goal scored (simplified - needs team checking)
            reward += self.reward_weights.get('goal_scored', 10.0)
        
        # Ball touch
        player = packet.game_cars[player_index]
        if player.ball_touched:
            reward += self.reward_weights.get('touch_ball', 0.5)
        
        # Boost pickup
        if prev_packet:
            prev_boost = prev_packet.game_cars[player_index].boost
            if player.boost > prev_boost:
                reward += self.reward_weights.get('boost_pickup', 0.1)
        
        return reward
