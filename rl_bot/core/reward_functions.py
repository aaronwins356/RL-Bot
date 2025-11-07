"""
Modular reward function components for Rocket League RL training.
Each component calculates a specific aspect of the reward signal.
"""

import numpy as np
from typing import Dict, Any, List
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions import RewardFunction


class CombinedReward(RewardFunction):
    """
    Combines multiple reward components with configurable weights.
    """
    
    def __init__(self, reward_functions: List[tuple], weights: Dict[str, float] = None):
        """
        Args:
            reward_functions: List of (name, reward_function) tuples
            weights: Dictionary mapping reward names to weights
        """
        super().__init__()
        self.reward_functions = dict(reward_functions)
        self.weights = weights or {}
        
    def reset(self, initial_state: GameState):
        """Reset all reward functions."""
        for reward_fn in self.reward_functions.values():
            reward_fn.reset(initial_state)
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate combined reward from all components."""
        total_reward = 0.0
        
        for name, reward_fn in self.reward_functions.items():
            weight = self.weights.get(name, 1.0)
            reward = reward_fn.get_reward(player, state, previous_action)
            total_reward += weight * reward
            
        return total_reward


class GoalReward(RewardFunction):
    """
    Reward for scoring goals and penalty for conceding.
    """
    
    def __init__(self, goal_reward: float = 10.0, concede_penalty: float = -10.0):
        super().__init__()
        self.goal_reward = goal_reward
        self.concede_penalty = concede_penalty
        self.last_score = None
    
    def reset(self, initial_state: GameState):
        """Reset score tracking."""
        self.last_score = None
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate goal-based reward."""
        if self.last_score is None:
            self.last_score = (state.blue_score, state.orange_score)
            return 0.0
        
        reward = 0.0
        current_score = (state.blue_score, state.orange_score)
        
        # Check if player's team scored
        if player.team_num == 0:  # Blue team
            if current_score[0] > self.last_score[0]:
                reward = self.goal_reward
            elif current_score[1] > self.last_score[1]:
                reward = self.concede_penalty
        else:  # Orange team
            if current_score[1] > self.last_score[1]:
                reward = self.goal_reward
            elif current_score[0] > self.last_score[0]:
                reward = self.concede_penalty
        
        self.last_score = current_score
        return reward


class TouchBallReward(RewardFunction):
    """
    Reward for touching the ball.
    """
    
    def __init__(self, touch_reward: float = 0.5):
        super().__init__()
        self.touch_reward = touch_reward
        self.last_touched = False
    
    def reset(self, initial_state: GameState):
        """Reset touch tracking."""
        self.last_touched = False
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate ball touch reward."""
        # Check if player touched the ball (simple distance-based heuristic)
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        distance = np.linalg.norm(ball_pos - player_pos)
        
        # Consider a touch if within 150 units and ball velocity changed
        touched = distance < 150
        
        reward = 0.0
        if touched and not self.last_touched:
            reward = self.touch_reward
        
        self.last_touched = touched
        return reward


class VelocityBallToGoalReward(RewardFunction):
    """
    Reward based on ball velocity toward opponent's goal.
    """
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def reset(self, initial_state: GameState):
        """No state to reset."""
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate velocity-based reward."""
        # Determine opponent goal position
        if player.team_num == 0:  # Blue team, attacking orange goal
            goal_pos = np.array([0, 5120, 0])  # Orange goal
        else:  # Orange team, attacking blue goal
            goal_pos = np.array([0, -5120, 0])  # Blue goal
        
        # Calculate ball direction to goal
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        to_goal = goal_pos - ball_pos
        to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)
        
        # Project ball velocity onto direction to goal
        vel_to_goal = np.dot(ball_vel, to_goal_norm)
        
        # Normalize and scale
        reward = self.weight * np.tanh(vel_to_goal / 2300)  # 2300 is max ball speed
        
        return reward


class AerialTouchReward(RewardFunction):
    """
    Bonus reward for aerial touches (ball and player both in air).
    """
    
    def __init__(self, aerial_reward: float = 1.0, min_height: float = 300):
        super().__init__()
        self.aerial_reward = aerial_reward
        self.min_height = min_height
        self.last_touched = False
    
    def reset(self, initial_state: GameState):
        """Reset touch tracking."""
        self.last_touched = False
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate aerial touch reward."""
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        
        # Check if both ball and player are in air
        ball_in_air = ball_pos[2] > self.min_height
        player_in_air = player_pos[2] > self.min_height and not player.on_ground
        
        # Check if player touched the ball
        distance = np.linalg.norm(ball_pos - player_pos)
        touched = distance < 150
        
        reward = 0.0
        if touched and not self.last_touched and ball_in_air and player_in_air:
            reward = self.aerial_reward
        
        self.last_touched = touched
        return reward


class SaveReward(RewardFunction):
    """
    Reward for making saves (clearing ball heading toward own goal).
    """
    
    def __init__(self, save_reward: float = 3.0):
        super().__init__()
        self.save_reward = save_reward
    
    def reset(self, initial_state: GameState):
        """No state to reset."""
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate save reward."""
        # Determine own goal position
        if player.team_num == 0:  # Blue team
            goal_pos = np.array([0, -5120, 0])
        else:  # Orange team
            goal_pos = np.array([0, 5120, 0])
        
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        player_pos = player.car_data.position
        
        # Check if ball is heading toward goal
        to_goal = goal_pos - ball_pos
        to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)
        vel_to_goal = np.dot(ball_vel, to_goal_norm)
        
        # Check if player touched ball
        distance = np.linalg.norm(ball_pos - player_pos)
        touched = distance < 150
        
        # Check if ball is close to goal
        ball_to_goal_dist = np.linalg.norm(to_goal)
        close_to_goal = ball_to_goal_dist < 2000
        
        # Reward if player touched ball that was heading to own goal
        reward = 0.0
        if touched and vel_to_goal > 500 and close_to_goal:
            reward = self.save_reward
        
        return reward


class BoostPickupReward(RewardFunction):
    """
    Reward for collecting boost pads.
    """
    
    def __init__(self, small_boost_reward: float = 0.05, large_boost_reward: float = 0.15):
        super().__init__()
        self.small_boost_reward = small_boost_reward
        self.large_boost_reward = large_boost_reward
        self.last_boost = 0
    
    def reset(self, initial_state: GameState):
        """Reset boost tracking."""
        self.last_boost = 0
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate boost pickup reward."""
        current_boost = player.boost_amount
        boost_gained = current_boost - self.last_boost
        
        reward = 0.0
        if boost_gained > 0:
            # Estimate if it was a small or large boost pad
            if boost_gained >= 100:
                reward = self.large_boost_reward
            else:
                reward = self.small_boost_reward
        
        self.last_boost = current_boost
        return reward


class FlickAttemptReward(RewardFunction):
    """
    Reward for attempting flicks (based on velocity and proximity).
    """
    
    def __init__(self, flick_reward: float = 0.3):
        super().__init__()
        self.flick_reward = flick_reward
    
    def reset(self, initial_state: GameState):
        """No state to reset."""
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate flick attempt reward."""
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        ball_vel = state.ball.linear_velocity
        
        # Check if ball is close and moving fast (potential flick)
        distance = np.linalg.norm(ball_pos - player_pos)
        ball_speed = np.linalg.norm(ball_vel)
        
        # Check if player is on ground (flicks happen on ground)
        on_ground = player.on_ground
        
        reward = 0.0
        if distance < 200 and ball_speed > 1500 and on_ground:
            reward = self.flick_reward
        
        return reward


class BumpAttemptReward(RewardFunction):
    """
    Reward for bumping/demoing opponents.
    """
    
    def __init__(self, bump_reward: float = 0.2, demo_reward: float = 2.0):
        super().__init__()
        self.bump_reward = bump_reward
        self.demo_reward = demo_reward
    
    def reset(self, initial_state: GameState):
        """No state to reset."""
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate bump/demo reward."""
        # Check if player is moving fast toward opponents
        player_pos = player.car_data.position
        player_vel = player.car_data.linear_velocity
        player_speed = np.linalg.norm(player_vel)
        
        reward = 0.0
        
        # Find closest opponent
        min_dist = float('inf')
        for other_player in state.players:
            if other_player.team_num != player.team_num:
                other_pos = other_player.car_data.position
                dist = np.linalg.norm(other_pos - player_pos)
                if dist < min_dist:
                    min_dist = dist
        
        # Reward if moving fast toward opponent (potential demo)
        if min_dist < 300 and player_speed > 2200:  # Supersonic speed
            reward = self.demo_reward
        elif min_dist < 200 and player_speed > 1400:  # Fast bump
            reward = self.bump_reward
        
        return reward


class PositioningReward(RewardFunction):
    """
    Reward for good positioning relative to ball and goal.
    Encourages defensive positioning when needed.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def reset(self, initial_state: GameState):
        """No state to reset."""
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate positioning reward."""
        # Determine own goal position
        if player.team_num == 0:  # Blue team
            own_goal = np.array([0, -5120, 0])
            opponent_goal = np.array([0, 5120, 0])
        else:  # Orange team
            own_goal = np.array([0, 5120, 0])
            opponent_goal = np.array([0, -5120, 0])
        
        player_pos = player.car_data.position
        ball_pos = state.ball.position
        
        # Calculate distances
        ball_to_own_goal = np.linalg.norm(ball_pos - own_goal)
        player_to_ball = np.linalg.norm(player_pos - ball_pos)
        player_to_own_goal = np.linalg.norm(player_pos - own_goal)
        
        # If ball is close to own goal, reward being between ball and goal
        if ball_to_own_goal < 3000:
            # Defensive positioning: be between ball and goal
            ideal_pos = own_goal + (ball_pos - own_goal) * 0.3
            dist_to_ideal = np.linalg.norm(player_pos - ideal_pos)
            reward = -dist_to_ideal / 3000  # Normalized
        else:
            # Offensive positioning: be closer to ball
            reward = -player_to_ball / 5000  # Normalized
        
        return reward * self.weight


class RotationReward(RewardFunction):
    """
    Reward for proper rotation in team play.
    Encourages not double-committing and maintaining spread.
    """
    
    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight
    
    def reset(self, initial_state: GameState):
        """No state to reset."""
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate rotation reward."""
        # Only relevant for team play (2v2, 3v3)
        teammates = [p for p in state.players if p.team_num == player.team_num and p.car_id != player.car_id]
        
        if len(teammates) == 0:
            return 0.0  # No rotation in 1v1
        
        player_pos = player.car_data.position
        ball_pos = state.ball.position
        
        # Calculate player distance to ball
        player_ball_dist = np.linalg.norm(player_pos - ball_pos)
        
        # Check if any teammate is closer to ball
        teammate_closer = False
        for teammate in teammates:
            tm_dist = np.linalg.norm(teammate.car_data.position - ball_pos)
            if tm_dist < player_ball_dist - 500:  # Teammate significantly closer
                teammate_closer = True
                break
        
        # If teammate is closer, reward being behind (rotational spread)
        if teammate_closer:
            # Reward being further from ball (rotating back)
            reward = player_ball_dist / 5000  # Normalized positive
        else:
            # Reward being closer to ball (front man)
            reward = -player_ball_dist / 5000  # Negative becomes positive when closest
        
        return reward * self.weight


def create_reward_function(config: Dict[str, Any]) -> RewardFunction:
    """
    Factory function to create combined reward function from config.
    
    Args:
        config: Configuration dictionary with reward weights
        
    Returns:
        Combined reward function
    """
    reward_weights = config.get('rewards', {})
    
    # Create individual reward components
    components = [
        ('goal', GoalReward(
            goal_reward=reward_weights.get('goal_scored', 10.0),
            concede_penalty=reward_weights.get('goal_conceded', -10.0)
        )),
        ('touch', TouchBallReward(
            touch_reward=reward_weights.get('touch_ball', 0.5)
        )),
        ('velocity_to_goal', VelocityBallToGoalReward(
            weight=reward_weights.get('velocity_ball_to_goal', 0.5)
        )),
        ('aerial_touch', AerialTouchReward(
            aerial_reward=reward_weights.get('touch_ball_aerial', 1.0),
            min_height=300
        )),
        ('save', SaveReward(
            save_reward=reward_weights.get('save', 3.0)
        )),
        ('boost_pickup', BoostPickupReward(
            small_boost_reward=reward_weights.get('boost_pickup', 0.1) * 0.5,
            large_boost_reward=reward_weights.get('boost_pickup', 0.1)
        )),
        ('flick', FlickAttemptReward(
            flick_reward=reward_weights.get('flick_attempt', 0.3)
        )),
        ('bump', BumpAttemptReward(
            bump_reward=reward_weights.get('bump_attempt', 0.2),
            demo_reward=reward_weights.get('demo', 2.0)
        )),
        ('positioning', PositioningReward(
            weight=reward_weights.get('positioning_weight', 0.1)
        )),
        ('rotation', RotationReward(
            weight=reward_weights.get('rotation_weight', 0.05)
        )),
    ]
    
    # Create combined reward
    return CombinedReward(components, weights=reward_weights)
