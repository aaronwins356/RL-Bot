"""Risk Scorer for opportunity detection."""

from typing import Dict, Any
import numpy as np


class RiskScorer:
    """Scores risk level based on game state for attempt selection.
    
    Heuristics from §8:
    - Score differential
    - Time remaining
    - Last defender status
    - Ball position (defensive third)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk scorer.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.safe_time_threshold = config.get('safe_time_threshold', 45.0)
        self.defensive_third_y = config.get('defensive_third_y', -3000)
        
    def compute_risk_score(self, obs: Dict[str, Any]) -> float:
        """Compute risk score (0-1, higher = more risky).
        
        Args:
            obs: Current observation
            
        Returns:
            Risk score between 0 and 1
        """
        risk_components = []
        
        # 1. Score differential risk
        score_diff = obs.get('score_diff', 0)
        if score_diff < 0:  # Losing
            risk_components.append(0.3)
        elif score_diff == 0:  # Tied
            risk_components.append(0.2)
        else:  # Winning
            risk_components.append(0.0)
        
        # 2. Time remaining risk
        time_remaining = obs.get('time_remaining', 300.0)
        if time_remaining < 30.0:
            risk_components.append(0.4)
        elif time_remaining < self.safe_time_threshold:
            risk_components.append(0.2)
        else:
            risk_components.append(0.0)
        
        # 3. Last defender risk
        is_last_defender = obs.get('is_last_defender', False)
        if is_last_defender:
            risk_components.append(0.3)
        else:
            risk_components.append(0.0)
        
        # 4. Ball in defensive third risk
        ball_pos = obs.get('ball_position', np.zeros(3))
        our_goal_y = obs.get('our_goal_y', -5120)
        
        # Check if ball is in defensive third
        defensive_third_threshold = our_goal_y + (5120 / 3)
        if abs(ball_pos[1] - our_goal_y) < abs(defensive_third_threshold - our_goal_y):
            risk_components.append(0.2)
        else:
            risk_components.append(0.0)
        
        # Compute total risk (average of components)
        total_risk = np.mean(risk_components)
        
        return np.clip(total_risk, 0.0, 1.0)
    
    def should_attempt_flashy(self, obs: Dict[str, Any]) -> bool:
        """Check if should attempt flashy mechanic.
        
        Heuristics from §8:
        - Possession probability ≥ 0.6
        - Opponent closest-ETA − own ETA ≥ 150ms
        - Boost ≥ 40 (≥ 30 for musty)
        - Score diff ≥ 0 OR time remaining ≥ 45s
        - Never when last defender and ball in defensive third
        
        Args:
            obs: Current observation
            
        Returns:
            True if should attempt flashy
        """
        # Check possession probability
        possession_prob = obs.get('possession_prob', 0.5)
        if possession_prob < 0.6:
            return False
        
        # Check opponent ETA advantage
        own_eta = obs.get('own_eta_to_ball', 1.0)
        opponent_eta = obs.get('opponent_closest_eta', 2.0)
        eta_advantage_ms = (opponent_eta - own_eta) * 1000
        if eta_advantage_ms < 150:
            return False
        
        # Check boost
        boost = obs.get('boost', 0)
        if boost < 40:  # General threshold (30 for musty specifically)
            return False
        
        # Check game state (score/time)
        score_diff = obs.get('score_diff', 0)
        time_remaining = obs.get('time_remaining', 300.0)
        if score_diff < 0 and time_remaining < 45.0:
            return False
        
        # Never when last defender in defensive third
        is_last_defender = obs.get('is_last_defender', False)
        ball_pos = obs.get('ball_position', np.zeros(3))
        our_goal_y = obs.get('our_goal_y', -5120)
        defensive_third_threshold = our_goal_y + (5120 / 3)
        in_defensive_third = abs(ball_pos[1] - our_goal_y) < abs(defensive_third_threshold - our_goal_y)
        
        if is_last_defender and in_defensive_third:
            return False
        
        return True
    
    def check_mechanic_feasibility(self, obs: Dict[str, Any], mechanic: str) -> bool:
        """Check if specific mechanic is feasible given game state.
        
        Args:
            obs: Current observation
            mechanic: Mechanic name (e.g., 'ceiling', 'flip_reset', 'musty')
            
        Returns:
            True if feasible
        """
        ball_pos = obs.get('ball_position', np.zeros(3))
        ball_vel = obs.get('ball_velocity', np.zeros(3))
        ball_height = ball_pos[2]
        
        if mechanic == 'ceiling':
            # Need ball near ceiling
            return ball_height > 1500
        
        elif mechanic == 'flip_reset':
            # Need ball in air at moderate height
            return 200 < ball_height < 1800
        
        elif mechanic == 'musty':
            # Need ball slightly above car
            car_pos = obs.get('car_position', np.zeros(3))
            relative_height = ball_height - car_pos[2]
            return 30 < relative_height < 150
        
        elif mechanic == 'breezi':
            # Need specific approach angle
            car_pos = obs.get('car_position', np.zeros(3))
            car_to_ball = ball_pos - car_pos
            distance = np.linalg.norm(car_to_ball)
            if distance < 1e-6:
                return False
            approach_angle = np.arctan2(car_to_ball[2], 
                                       np.linalg.norm(car_to_ball[:2])) * 180 / np.pi
            return 20 <= approach_angle <= 35
        
        elif mechanic == 'double_tap':
            # Need ball heading toward backboard
            return ball_vel[1] > 500  # Moving toward opponent backboard
        
        elif mechanic == 'ground_to_air_dribble':
            # Need ball on ground or low
            return ball_height < 200
        
        return True
