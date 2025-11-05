"""
Utility-Based Decision System
Makes high-level strategic decisions based on game state evaluation.
Complements the neural network with explicit SSL-level game sense.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum

from util.game_state import GameState
from util.player_data import PlayerData
from util.ball_prediction import BallPredictor
from util.boost_manager import BoostManager


class BehaviorType(Enum):
    """High-level behavior categories"""
    ATTACK = "attack"
    DEFEND = "defend"
    SHADOW = "shadow"
    COLLECT_BOOST = "collect_boost"
    AERIAL = "aerial"
    DRIBBLE = "dribble"
    CHALLENGE = "challenge"
    WAIT = "wait"
    RECOVER = "recover"


class GameSituation:
    """Analyzes current game situation"""
    def __init__(self, player: PlayerData, opponent: PlayerData, 
                 game_state: GameState, ball_predictor: BallPredictor,
                 boost_manager: BoostManager):
        self.player = player
        self.opponent = opponent
        self.game_state = game_state
        self.ball_predictor = ball_predictor
        self.boost_manager = boost_manager
        
        # Calculate key metrics
        self._analyze_situation()
    
    def _analyze_situation(self):
        """Compute situation metrics"""
        ball = self.game_state.ball
        car = self.player.car_data
        opp_car = self.opponent.car_data
        
        # Goal positions (blue: y=-5120, orange: y=5120)
        our_goal_y = -5120 if self.player.team_num == 0 else 5120
        opp_goal_y = 5120 if self.player.team_num == 0 else -5120
        
        our_goal_pos = np.array([0, our_goal_y, 100])
        opp_goal_pos = np.array([0, opp_goal_y, 100])
        
        # Distances
        self.ball_dist = np.linalg.norm(ball.position - car.position)
        self.opp_ball_dist = np.linalg.norm(ball.position - opp_car.position)
        self.ball_to_our_goal = np.linalg.norm(ball.position - our_goal_pos)
        self.ball_to_opp_goal = np.linalg.norm(ball.position - opp_goal_pos)
        self.car_to_our_goal = np.linalg.norm(car.position - our_goal_pos)
        
        # Field position analysis
        self.ball_in_our_half = (ball.position[1] < 0) if self.player.team_num == 0 else (ball.position[1] > 0)
        self.ball_in_opp_half = not self.ball_in_our_half
        self.we_closer_to_ball = self.ball_dist < self.opp_ball_dist
        
        # Boost analysis
        self.our_boost = self.player.boost_amount
        self.opp_boost = self.opponent.boost_amount
        self.boost_advantage = self.our_boost - self.opp_boost
        self.low_boost = self.our_boost < 0.3
        
        # Ball state analysis
        self.ball_speed = np.linalg.norm(ball.linear_velocity)
        self.ball_on_ground = ball.position[2] < 150
        
        # Predict ball trajectory
        self.ball_landing_time = self.ball_predictor.get_landing_time(ball)
        self.ball_going_to_our_goal = self.ball_predictor.is_ball_going_towards_goal(
            ball, our_goal_pos, time_horizon=2.0
        )
        
        # Positioning analysis
        self.between_ball_and_goal = self._is_between(car.position, ball.position, our_goal_pos)
        self.good_defensive_position = self.between_ball_and_goal and self.car_to_our_goal < 3000
        
        # Opponent analysis
        opp_vel = opp_car.linear_velocity
        opp_to_ball = ball.position - opp_car.position
        self.opponent_attacking = np.dot(opp_vel, opp_to_ball) > 500  # Moving toward ball quickly
        
    def _is_between(self, pos: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> bool:
        """Check if pos is between point_a and point_b"""
        # Simple check: distance from pos to both points should be less than distance between points
        dist_a = np.linalg.norm(pos - point_a)
        dist_b = np.linalg.norm(pos - point_b)
        dist_ab = np.linalg.norm(point_a - point_b)
        
        return (dist_a + dist_b) < (dist_ab * 1.2)  # Allow 20% margin


class UtilitySystem:
    """
    Evaluates game situations and chooses optimal behavior.
    Uses utility scoring to make SSL-level strategic decisions.
    """
    
    def __init__(self):
        self.current_behavior = BehaviorType.WAIT
        self.behavior_history = []
        
    def evaluate(self, player: PlayerData, opponent: PlayerData,
                game_state: GameState, ball_predictor: BallPredictor,
                boost_manager: BoostManager) -> BehaviorType:
        """
        Evaluate current situation and choose best behavior.
        
        Returns:
            BehaviorType indicating what the bot should do
        """
        situation = GameSituation(player, opponent, game_state, ball_predictor, boost_manager)
        
        # Calculate utility scores for each behavior
        scores = {
            BehaviorType.ATTACK: self._score_attack(situation),
            BehaviorType.DEFEND: self._score_defend(situation),
            BehaviorType.SHADOW: self._score_shadow(situation),
            BehaviorType.COLLECT_BOOST: self._score_collect_boost(situation),
            BehaviorType.AERIAL: self._score_aerial(situation),
            BehaviorType.CHALLENGE: self._score_challenge(situation),
            BehaviorType.WAIT: self._score_wait(situation),
            BehaviorType.RECOVER: self._score_recover(situation),
        }
        
        # Choose behavior with highest score
        best_behavior = max(scores.items(), key=lambda x: x[1])[0]
        
        # Track behavior history
        self.behavior_history.append(best_behavior)
        if len(self.behavior_history) > 60:  # Keep last 60 decisions (~2 seconds)
            self.behavior_history.pop(0)
        
        self.current_behavior = best_behavior
        return best_behavior
    
    def _score_attack(self, sit: GameSituation) -> float:
        """Score attacking behavior"""
        score = 0.0
        
        # High score when:
        # - Ball in opponent half
        if sit.ball_in_opp_half:
            score += 30.0
        
        # - We're closer to ball than opponent
        if sit.we_closer_to_ball:
            score += 20.0
        
        # - We have boost
        score += sit.our_boost * 20.0
        
        # - Ball is close
        if sit.ball_dist < 1500:
            score += 20.0
        
        # - Opponent is far from ball
        if sit.opp_ball_dist > 2000:
            score += 15.0
        
        # - Ball is on ground (easier to control)
        if sit.ball_on_ground:
            score += 10.0
        
        # Low score when:
        # - Low boost
        if sit.low_boost:
            score -= 20.0
        
        # - Ball in our half (risky to attack)
        if sit.ball_in_our_half:
            score -= 15.0
        
        return score
    
    def _score_defend(self, sit: GameSituation) -> float:
        """Score defensive behavior"""
        score = 0.0
        
        # High score when:
        # - Ball in our half
        if sit.ball_in_our_half:
            score += 35.0
        
        # - Ball going toward our goal
        if sit.ball_going_to_our_goal:
            score += 40.0
        
        # - Opponent closer to ball
        if not sit.we_closer_to_ball:
            score += 25.0
        
        # - Opponent attacking
        if sit.opponent_attacking:
            score += 20.0
        
        # - Not in good defensive position yet
        if not sit.good_defensive_position:
            score += 15.0
        
        return score
    
    def _score_shadow(self, sit: GameSituation) -> float:
        """Score shadow defense (stay between ball and goal, don't commit)"""
        score = 0.0
        
        # High score when:
        # - Ball in opponent's possession (they're closer)
        if sit.opp_ball_dist < sit.ball_dist:
            score += 30.0
        
        # - Ball in midfield or our half
        if sit.ball_in_our_half or abs(sit.game_state.ball.position[1]) < 2000:
            score += 25.0
        
        # - We're already in good position
        if sit.good_defensive_position:
            score += 20.0
        
        # - Opponent has boost advantage
        if sit.boost_advantage < -0.2:
            score += 15.0
        
        # - We have some boost (need it to shadow effectively)
        if sit.our_boost > 0.3:
            score += 10.0
        
        # Low score when:
        # - Opponent very far from ball (no threat)
        if sit.opp_ball_dist > 3000:
            score -= 20.0
        
        # - Ball is slow and close (we should attack)
        if sit.ball_speed < 500 and sit.ball_dist < 1000:
            score -= 15.0
        
        return score
    
    def _score_collect_boost(self, sit: GameSituation) -> float:
        """Score boost collection behavior"""
        score = 0.0
        
        # High score when:
        # - Very low boost
        if sit.our_boost < 0.2:
            score += 40.0
        elif sit.our_boost < 0.4:
            score += 20.0
        
        # - Safe situation (ball far, opponent not attacking)
        if sit.ball_dist > 2500 and not sit.opponent_attacking:
            score += 25.0
        
        # - Boost disadvantage
        if sit.boost_advantage < -0.3:
            score += 20.0
        
        # Low score when:
        # - Already have boost
        if sit.our_boost > 0.7:
            score -= 50.0
        
        # - Ball is close (shouldn't leave it)
        if sit.ball_dist < 1500:
            score -= 20.0
        
        # - Defensive situation (need to defend, not collect boost)
        if sit.ball_going_to_our_goal:
            score -= 30.0
        
        return score
    
    def _score_aerial(self, sit: GameSituation) -> float:
        """Score aerial behavior"""
        score = 0.0
        
        # High score when:
        # - Ball is in air and high
        ball_height = sit.game_state.ball.position[2]
        if ball_height > 500:
            score += 30.0
        elif ball_height > 300:
            score += 15.0
        
        # - We have boost
        if sit.our_boost > 0.5:
            score += 20.0
        elif sit.our_boost > 0.3:
            score += 10.0
        
        # - We're close enough to reach
        if sit.ball_dist < 2000:
            score += 15.0
        
        # - Ball is somewhat slow (easier aerial)
        if sit.ball_speed < 1500:
            score += 10.0
        
        # Low score when:
        # - Low boost (can't sustain aerial)
        if sit.our_boost < 0.3:
            score -= 40.0
        
        # - Ball on ground
        if sit.ball_on_ground:
            score -= 50.0
        
        # - Too far to reach
        if sit.ball_dist > 3000:
            score -= 30.0
        
        return score
    
    def _score_challenge(self, sit: GameSituation) -> float:
        """Score challenging the ball (commit to 50/50)"""
        score = 0.0
        
        # High score when:
        # - Close to ball
        if sit.ball_dist < 1000:
            score += 30.0
        
        # - Opponent also going for ball (50/50 situation)
        if sit.opp_ball_dist < 1500 and sit.we_closer_to_ball:
            score += 25.0
        
        # - We have boost advantage
        if sit.boost_advantage > 0.2:
            score += 15.0
        
        # - Ball is contested area
        if abs(sit.game_state.ball.position[1]) < 3000:
            score += 10.0
        
        # Low score when:
        # - We're much farther than opponent
        if sit.ball_dist > sit.opp_ball_dist * 1.5:
            score -= 30.0
        
        # - Low boost
        if sit.low_boost:
            score -= 20.0
        
        # - Bad defensive position (risky to commit)
        if sit.ball_in_our_half and not sit.good_defensive_position:
            score -= 25.0
        
        return score
    
    def _score_wait(self, sit: GameSituation) -> float:
        """Score waiting/positioning behavior"""
        score = 0.0
        
        # High score when:
        # - In good position already
        if sit.good_defensive_position:
            score += 20.0
        
        # - Ball is far
        if sit.ball_dist > 2500:
            score += 15.0
        
        # - Opponent has ball (let them make mistake)
        if sit.opp_ball_dist < sit.ball_dist * 0.7:
            score += 15.0
        
        # - We have good boost
        if sit.our_boost > 0.5:
            score += 10.0
        
        # Low score when:
        # - Ball is close (should do something)
        if sit.ball_dist < 1500:
            score -= 20.0
        
        # - Ball going to our goal (can't wait)
        if sit.ball_going_to_our_goal:
            score -= 40.0
        
        # - Low boost (should collect)
        if sit.low_boost:
            score -= 15.0
        
        return score
    
    def _score_recover(self, sit: GameSituation) -> float:
        """Score recovery behavior (get back to good state)"""
        score = 0.0
        
        # High score when:
        # - Not on ground
        if not sit.player.on_ground:
            score += 40.0
        
        # - Bad orientation (upside down, etc.)
        car = sit.player.car_data
        up_vector = car.up()
        if up_vector[2] < 0.5:  # Not upright
            score += 30.0
        
        # - Far from action
        if sit.ball_dist > 3000:
            score += 10.0
        
        return score
    
    def get_behavior_info(self) -> Dict:
        """Get information about current behavior and decision-making"""
        return {
            'current_behavior': self.current_behavior,
            'behavior_history': self.behavior_history[-10:] if self.behavior_history else []
        }
