"""
Advanced observation builders for team play and ball prediction.
Extends the base observation with teammate/opponent info and predictions.
"""

import numpy as np
from typing import Optional
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rl_bot.core.ball_prediction import SimpleBallPredictor, PredictionFeatureExtractor


class TeamAwareObsBuilder(ObsBuilder):
    """
    Observation builder that includes teammate and opponent information.
    Supports 1v1, 2v2, and 3v3 with padding for variable team sizes.
    """
    
    def __init__(
        self,
        max_team_size: int = 3,
        include_predictions: bool = True,
        num_predictions: int = 5
    ):
        """
        Args:
            max_team_size: Maximum players per team (3 for 3v3)
            include_predictions: Include ball prediction features
            num_predictions: Number of future ball positions to include
        """
        super().__init__()
        self.max_team_size = max_team_size
        self.include_predictions = include_predictions
        self.num_predictions = num_predictions
        
        # Initialize predictor if needed
        if self.include_predictions:
            self.predictor = SimpleBallPredictor()
            self.feature_extractor = PredictionFeatureExtractor(self.predictor)
        
        # Calculate observation dimensions
        self._calculate_obs_dim()
    
    def _calculate_obs_dim(self):
        """Calculate total observation dimension."""
        # Own player: position(3) + velocity(3) + rotation(9) + angular_vel(3) + boost(1) + on_ground(1) = 20
        player_dim = 20
        
        # Ball: position(3) + velocity(3) + angular_vel(3) = 9
        ball_dim = 9
        
        # Relative to ball: pos_rel(3) + vel_rel(3) = 6
        relative_dim = 6
        
        # Other players (teammates + opponents): per player: position(3) + velocity(3) + relative_pos(3) = 9
        # Max of (max_team_size - 1) teammates + max_team_size opponents
        max_other_players = (self.max_team_size - 1) + self.max_team_size
        other_players_dim = max_other_players * 9
        
        # Ball predictions: num_predictions * 3 (position only)
        prediction_dim = self.num_predictions * 3 if self.include_predictions else 0
        
        self.obs_dim = player_dim + ball_dim + relative_dim + other_players_dim + prediction_dim
    
    def reset(self, initial_state: GameState):
        """Reset observation builder."""
        pass
    
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Build observation including teammates and opponents.
        
        Args:
            player: Current player data
            state: Current game state
            previous_action: Previous action taken
            
        Returns:
            Flat observation array
        """
        obs = []
        
        # 1. Own player data
        car = player.car_data
        obs.extend(car.position / 4096)
        obs.extend(car.linear_velocity / 2300)
        obs.extend(car.rotation_mtx().flatten())
        obs.extend(car.angular_velocity / 5.5)
        obs.append(player.boost_amount / 100)
        obs.append(float(player.on_ground))
        
        # 2. Ball data
        ball = state.ball
        obs.extend(ball.position / 4096)
        obs.extend(ball.linear_velocity / 6000)
        obs.extend(ball.angular_velocity / 6)
        
        # 3. Relative ball information
        ball_rel_pos = (ball.position - car.position) / 4096
        ball_rel_vel = (ball.linear_velocity - car.linear_velocity) / 2300
        obs.extend(ball_rel_pos)
        obs.extend(ball_rel_vel)
        
        # 4. Teammates and opponents
        teammates = []
        opponents = []
        
        for other_player in state.players:
            if other_player.car_id == player.car_id:
                continue  # Skip self
            
            if other_player.team_num == player.team_num:
                teammates.append(other_player)
            else:
                opponents.append(other_player)
        
        # Add teammate data (with padding)
        max_teammates = self.max_team_size - 1
        for i in range(max_teammates):
            if i < len(teammates):
                tm = teammates[i]
                obs.extend(tm.car_data.position / 4096)
                obs.extend(tm.car_data.linear_velocity / 2300)
                rel_pos = (tm.car_data.position - car.position) / 4096
                obs.extend(rel_pos)
            else:
                # Padding: 9 zeros
                obs.extend([0] * 9)
        
        # Add opponent data (with padding)
        for i in range(self.max_team_size):
            if i < len(opponents):
                opp = opponents[i]
                obs.extend(opp.car_data.position / 4096)
                obs.extend(opp.car_data.linear_velocity / 2300)
                rel_pos = (opp.car_data.position - car.position) / 4096
                obs.extend(rel_pos)
            else:
                # Padding: 9 zeros
                obs.extend([0] * 9)
        
        # 5. Ball predictions
        if self.include_predictions:
            pred_features = self.feature_extractor.get_prediction_features(
                ball.position,
                ball.linear_velocity,
                ball.angular_velocity,
                car.position,
                num_predictions=self.num_predictions
            )
            obs.extend(pred_features)
        
        return np.asarray(obs, dtype=np.float32)


class CompactObsBuilder(ObsBuilder):
    """
    Compact observation builder for faster training.
    Includes only essential information with ball predictions.
    """
    
    def __init__(self, include_predictions: bool = True):
        """
        Args:
            include_predictions: Include ball prediction features
        """
        super().__init__()
        self.include_predictions = include_predictions
        
        if self.include_predictions:
            self.predictor = SimpleBallPredictor()
            self.feature_extractor = PredictionFeatureExtractor(self.predictor)
        
        # Compact dimensions:
        # Player: pos(3) + vel(3) + forward(3) + boost(1) + on_ground(1) = 11
        # Ball: pos(3) + vel(3) = 6
        # Relative: ball_rel_pos(3) + ball_rel_vel(3) = 6
        # Predictions: 5 * 3 = 15 (if enabled)
        self.obs_dim = 11 + 6 + 6 + (15 if include_predictions else 0)
    
    def reset(self, initial_state: GameState):
        """Reset observation builder."""
        pass
    
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Build compact observation.
        
        Args:
            player: Current player data
            state: Current game state
            previous_action: Previous action taken
            
        Returns:
            Compact observation array
        """
        obs = []
        
        # Player essentials
        car = player.car_data
        obs.extend(car.position / 4096)
        obs.extend(car.linear_velocity / 2300)
        obs.extend(car.forward())  # Forward direction vector
        obs.append(player.boost_amount / 100)
        obs.append(float(player.on_ground))
        
        # Ball essentials
        ball = state.ball
        obs.extend(ball.position / 4096)
        obs.extend(ball.linear_velocity / 6000)
        
        # Relative
        ball_rel_pos = (ball.position - car.position) / 4096
        ball_rel_vel = (ball.linear_velocity - car.linear_velocity) / 2300
        obs.extend(ball_rel_pos)
        obs.extend(ball_rel_vel)
        
        # Predictions
        if self.include_predictions:
            pred_features = self.feature_extractor.get_prediction_features(
                ball.position,
                ball.linear_velocity,
                ball.angular_velocity,
                car.position,
                num_predictions=5
            )
            obs.extend(pred_features)
        
        return np.asarray(obs, dtype=np.float32)
