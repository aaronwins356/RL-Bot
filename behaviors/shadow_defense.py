"""
Shadow Defense Behavior
Implements SSL-level shadow defense positioning and fake challenges.
Key 1v1 skill: stay between ball and goal without committing.
"""

import numpy as np
from typing import List, Tuple, Optional

from util.game_state import GameState
from util.player_data import PlayerData
from util.ball_prediction import BallPredictor


class ShadowDefense:
    """
    Shadow defense: Position between ball and goal, mirror opponent's movement,
    wait for opportunity to challenge or counter-attack.
    
    SSL players use this to:
    - Apply pressure without committing
    - Force opponent into mistakes
    - Maintain defensive positioning
    - Create fake challenge opportunities
    """
    
    def __init__(self):
        self.min_shadow_distance = 800    # Don't get closer than this
        self.ideal_shadow_distance = 1200  # Ideal distance to maintain
        self.max_shadow_distance = 2000   # Don't fall back further than this
        self.fake_challenge_cooldown = 0.0
        self.last_fake_time = 0.0
        
    def calculate_shadow_position(self, player: PlayerData, opponent: PlayerData,
                                  game_state: GameState) -> np.ndarray:
        """
        Calculate optimal shadow position.
        
        Returns:
            Target position to maintain shadow defense
        """
        ball = game_state.ball
        car_pos = player.car_data.position
        
        # Determine our goal position
        our_goal_y = -5120 if player.team_num == 0 else 5120
        our_goal_pos = np.array([0, our_goal_y, 100])
        
        # Vector from ball to goal
        ball_to_goal = our_goal_pos - ball.position
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)
        
        if ball_to_goal_dist < 1:
            # Ball is basically at goal, defend goal line
            return our_goal_pos
        
        ball_to_goal_norm = ball_to_goal / ball_to_goal_dist
        
        # Position ourselves between ball and goal
        # Distance depends on ball proximity to goal
        ball_danger = max(0, (6000 - ball_to_goal_dist) / 6000)  # 0 (far) to 1 (close)
        shadow_distance = self.ideal_shadow_distance * (1 - ball_danger * 0.5)
        shadow_distance = np.clip(shadow_distance, self.min_shadow_distance, self.max_shadow_distance)
        
        # Shadow position is along the ball-to-goal line
        shadow_pos = ball.position + ball_to_goal_norm * shadow_distance
        
        # Adjust x position to mirror opponent's x (stay between opponent and goal)
        opp_pos = opponent.car_data.position
        
        # If opponent is to the side, adjust our x to block their angle
        ball_to_opp = opp_pos - ball.position
        if abs(ball_to_opp[0]) > 500:  # Opponent is significantly to the side
            # Shift shadow position to block angle
            shadow_pos[0] = ball.position[0] + np.sign(ball_to_opp[0]) * 300
        
        # Keep shadow position on field
        shadow_pos[0] = np.clip(shadow_pos[0], -4000, 4000)
        shadow_pos[1] = np.clip(shadow_pos[1], -5000, 5000)
        shadow_pos[2] = 0  # Ground level
        
        return shadow_pos
    
    def should_challenge(self, player: PlayerData, opponent: PlayerData,
                        game_state: GameState, ball_predictor: BallPredictor) -> bool:
        """
        Decide if we should break shadow and commit to challenge.
        
        SSL players challenge when:
        - Opponent has bad touch (ball too far ahead)
        - Ball is slow and close
        - Opponent low on boost
        - We have boost advantage
        - Ball bouncing awkwardly
        
        Returns:
            True if should challenge now
        """
        ball = game_state.ball
        car_pos = player.car_data.position
        opp_pos = opponent.car_data.position
        
        # Distances
        ball_dist = np.linalg.norm(ball.position - car_pos)
        opp_ball_dist = np.linalg.norm(ball.position - opp_pos)
        
        # Ball state
        ball_speed = np.linalg.norm(ball.linear_velocity)
        ball_height = ball.position[2]
        
        # Challenge conditions
        challenge_score = 0
        
        # 1. Opponent has bad touch (ball far from them)
        if opp_ball_dist > 800:
            challenge_score += 30
        
        # 2. Ball is slow (easier to control)
        if ball_speed < 800:
            challenge_score += 20
        
        # 3. We're close enough to reach first
        if ball_dist < 1000 and ball_dist < opp_ball_dist * 0.8:
            challenge_score += 25
        
        # 4. Boost advantage
        if player.boost_amount > opponent.boost_amount + 0.2:
            challenge_score += 15
        
        # 5. Ball on ground (easier challenge)
        if ball_height < 200:
            challenge_score += 10
        
        # 6. Ball bouncing (opponent may miss)
        landing_time = ball_predictor.get_landing_time(ball)
        if landing_time is not None and landing_time < 0.5:
            challenge_score += 15
        
        # Don't challenge if:
        # - We're too far
        if ball_dist > 1500:
            challenge_score -= 40
        
        # - Opponent is supersonic toward ball
        opp_vel = opponent.car_data.linear_velocity
        opp_speed = np.linalg.norm(opp_vel)
        if opp_speed > 2200 and opp_ball_dist < 1000:
            challenge_score -= 30
        
        # - Very low boost
        if player.boost_amount < 0.15:
            challenge_score -= 20
        
        return challenge_score > 40
    
    def should_fake_challenge(self, player: PlayerData, opponent: PlayerData,
                             game_state: GameState, current_time: float) -> bool:
        """
        Decide if we should fake challenge (drive at ball then dodge away).
        
        Fake challenges:
        - Force opponent to rush their touch
        - Create hesitation and mistakes
        - Maintain pressure without committing
        
        Returns:
            True if should execute fake challenge
        """
        # Cooldown check (don't spam fake challenges)
        time_since_last_fake = current_time - self.last_fake_time
        if time_since_last_fake < 3.0:  # 3 second cooldown
            return False
        
        ball = game_state.ball
        car_pos = player.car_data.position
        opp_pos = opponent.car_data.position
        
        ball_dist = np.linalg.norm(ball.position - car_pos)
        opp_ball_dist = np.linalg.norm(ball.position - opp_pos)
        
        # Fake challenge conditions
        fake_score = 0
        
        # 1. Opponent has ball (close to it)
        if opp_ball_dist < 600:
            fake_score += 30
        
        # 2. We're in good position (not too far, not too close)
        if 1000 < ball_dist < 1800:
            fake_score += 25
        
        # 3. Opponent is dribbling (slow ball)
        ball_speed = np.linalg.norm(ball.linear_velocity)
        if ball_speed < 500 and opp_ball_dist < 300:
            fake_score += 20
        
        # 4. We have boost to execute
        if player.boost_amount > 0.4:
            fake_score += 15
        
        # 5. Opponent is facing us (will see the fake)
        opp_to_us = car_pos - opp_pos
        opp_forward = opponent.car_data.forward()
        facing_us = np.dot(opp_forward, opp_to_us) > 0
        if facing_us:
            fake_score += 10
        
        return fake_score > 60
    
    def execute_fake_challenge(self, player: PlayerData, game_state: GameState,
                               current_time: float) -> Tuple[bool, List[float]]:
        """
        Execute fake challenge maneuver.
        
        Returns:
            (is_executing, controls) tuple
        """
        # Check if we should start fake challenge
        if self.fake_challenge_cooldown <= 0:
            # Start fake challenge
            self.fake_challenge_cooldown = 0.5  # 0.5 second duration
            self.last_fake_time = current_time
        
        if self.fake_challenge_cooldown > 0:
            self.fake_challenge_cooldown -= 1/30  # Assume 30 FPS
            
            ball = game_state.ball
            car = player.car_data
            
            # Phase 1 (first 0.3s): Drive toward ball aggressively
            if self.fake_challenge_cooldown > 0.2:
                to_ball = ball.position - car.position
                to_ball_norm = to_ball / (np.linalg.norm(to_ball) + 0.001)
                
                # Point toward ball
                forward = car.forward()
                steer = np.cross(forward, to_ball_norm)[2] * 3.0
                steer = np.clip(steer, -1, 1)
                
                return True, [1.0, steer, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Full throttle + boost
            
            # Phase 2 (last 0.2s): Dodge backward/away
            else:
                # Dodge backward
                return True, [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # Backflip away
        
        return False, []
    
    def calculate_shadow_controls(self, player: PlayerData, shadow_target: np.ndarray,
                                  game_state: GameState) -> List[float]:
        """
        Calculate controls to maintain shadow position.
        
        Returns:
            Control values [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        car = player.car_data
        ball = game_state.ball
        
        # Vector to shadow target
        to_target = shadow_target - car.position
        distance = np.linalg.norm(to_target)
        
        if distance < 10:
            # At target, just face ball
            to_ball = ball.position - car.position
            to_ball_norm = to_ball / (np.linalg.norm(to_ball) + 0.001)
            forward = car.forward()
            steer = np.cross(forward, to_ball_norm)[2] * 3.0
            steer = np.clip(steer, -1, 1)
            return [0.0, steer, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        to_target_norm = to_target / distance
        
        # Calculate steering to point toward target
        forward = car.forward()
        steer = np.cross(forward, to_target_norm)[2] * 3.0
        steer = np.clip(steer, -1, 1)
        
        # Calculate throttle based on distance
        if distance > 1000:
            throttle = 1.0
            boost = 1.0 if distance > 1500 else 0.0
        elif distance > 300:
            throttle = 0.7
            boost = 0.0
        else:
            throttle = 0.3
            boost = 0.0
        
        # Face ball while driving
        to_ball = ball.position - car.position
        to_ball_norm = to_ball / (np.linalg.norm(to_ball) + 0.001)
        
        # Blend steering between target and ball
        ball_facing = np.cross(forward, to_ball_norm)[2] * 1.5
        steer = steer * 0.7 + ball_facing * 0.3
        steer = np.clip(steer, -1, 1)
        
        return [throttle, steer, 0.0, 0.0, 0.0, 0.0, boost, 0.0]
    
    def reset(self):
        """Reset fake challenge cooldown"""
        self.fake_challenge_cooldown = 0.0
