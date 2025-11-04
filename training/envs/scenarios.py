from typing import List, Dict, Any
import numpy as np
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.gamestates import GameState

class AerialInterceptScenario:
    class ObsBuilder(ObsBuilder):
        def reset(self, initial_state: GameState):
            pass

        def build_obs(self, player: int, state: GameState, previous_action: np.ndarray) -> np.ndarray:
            # Build observation focused on aerial position/velocity
            obs = []
            
            # Relative ball position and velocity
            rel_ball_pos = state.ball.position - state.players[player].car_data.position
            rel_ball_vel = state.ball.linear_velocity - state.players[player].car_data.linear_velocity
            
            obs.extend(rel_ball_pos)
            obs.extend(rel_ball_vel)
            
            # Car state
            obs.extend(state.players[player].car_data.position)
            obs.extend(state.players[player].car_data.linear_velocity)
            obs.extend(state.players[player].car_data.angular_velocity)
            obs.append(state.players[player].boost_amount)
            
            return np.array(obs)

    class Reward(RewardFunction):
        def reset(self, initial_state: GameState):
            self.last_touch_time = 0
            
        def get_reward(self, player: int, state: GameState, previous_action: np.ndarray) -> float:
            reward = 0
            
            # Reward for getting close to ball
            ball_dist = np.linalg.norm(
                state.ball.position - state.players[player].car_data.position
            )
            reward -= ball_dist * 0.1
            
            # Reward for aerial height
            car_height = state.players[player].car_data.position[2]
            if car_height > 100:
                reward += car_height * 0.01
                
            # Big reward for touching ball
            if state.last_touch == player:
                touch_time = state.game_info.seconds_elapsed
                if touch_time > self.last_touch_time:
                    reward += 10.0
                    self.last_touch_time = touch_time
                    
            return reward

    class Terminal(TerminalCondition):
        def reset(self, initial_state: GameState):
            self.timer = 0
            
        def is_terminal(self, current_state: GameState) -> bool:
            self.timer += 1
            
            # End if ball touched or timeout
            return (
                current_state.last_touch is not None or 
                self.timer >= 300
            )

    class State(StateSetter):
        def reset(self, state_wrapper: GameState):
            # Set up aerial scenario
            state_wrapper.ball.position = np.array([0, 0, 1000])
            state_wrapper.ball.linear_velocity = np.array([0, 0, 0])
            
            # Random starting position for car
            x = np.random.uniform(-1000, 1000)
            y = np.random.uniform(-1000, 1000)
            state_wrapper.players[0].car_data.position = np.array([x, y, 100])
            state_wrapper.players[0].boost_amount = 100


class WallCarryScenario:
    class ObsBuilder(ObsBuilder):
        def reset(self, initial_state: GameState):
            pass

        def build_obs(self, player: int, state: GameState, previous_action: np.ndarray) -> np.ndarray:
            obs = []
            
            # Distance to nearest wall
            car_pos = state.players[player].car_data.position
            wall_distances = [
                4096 - abs(car_pos[0]),  # Side walls
                5120 - abs(car_pos[1]),  # Back walls
            ]
            obs.append(min(wall_distances))
            
            # Ball and car state
            rel_ball_pos = state.ball.position - car_pos
            obs.extend(rel_ball_pos)
            obs.extend(state.ball.linear_velocity)
            obs.extend(car_pos)
            obs.extend(state.players[player].car_data.linear_velocity)
            
            return np.array(obs)

    class Reward(RewardFunction):
        def reset(self, initial_state: GameState):
            self.last_touch_time = 0
            self.on_wall = False
            
        def get_reward(self, player: int, state: GameState, previous_action: np.ndarray) -> float:
            reward = 0
            
            # Reward wall proximity
            car_pos = state.players[player].car_data.position
            wall_dist = min([
                4096 - abs(car_pos[0]),  # Side walls
                5120 - abs(car_pos[1]),  # Back walls
            ])
            
            if wall_dist < 100:
                reward += 1.0
                self.on_wall = True
            elif self.on_wall:
                # Penalize leaving wall without ball
                if state.last_touch != player:
                    reward -= 2.0
                self.on_wall = False
                
            # Reward ball control
            if state.last_touch == player:
                touch_time = state.game_info.seconds_elapsed
                if touch_time > self.last_touch_time:
                    reward += 5.0
                    self.last_touch_time = touch_time
                    
            return reward

    class Terminal(TerminalCondition):
        def reset(self, initial_state: GameState):
            self.timer = 0
            
        def is_terminal(self, current_state: GameState) -> bool:
            self.timer += 1
            return self.timer >= 600  # 20 seconds

    class State(StateSetter):
        def reset(self, state_wrapper: GameState):
            # Place car near wall
            x = 4000 * np.random.choice([-1, 1])
            y = np.random.uniform(-5000, 5000)
            state_wrapper.players[0].car_data.position = np.array([x, y, 100])
            
            # Place ball slightly away from wall
            ball_x = x * 0.8
            state_wrapper.ball.position = np.array([ball_x, y, 100])
            state_wrapper.players[0].boost_amount = 100


class FlipResetScenario:
    class ObsBuilder(ObsBuilder):
        def reset(self, initial_state: GameState):
            pass

        def build_obs(self, player: int, state: GameState, previous_action: np.ndarray) -> np.ndarray:
            obs = []
            
            # Car state including orientation
            car_state = state.players[player].car_data
            obs.extend(car_state.position)
            obs.extend(car_state.linear_velocity)
            obs.extend(car_state.angular_velocity)
            obs.extend(car_state.rotation_matrix.flatten())
            
            # Ball state
            rel_ball_pos = state.ball.position - car_state.position
            obs.extend(rel_ball_pos)
            obs.extend(state.ball.linear_velocity)
            
            # Additional flip reset specific features
            wheels_on_ball = self._check_wheels_contact(state, player)
            obs.append(float(wheels_on_ball))
            
            return np.array(obs)
            
        def _check_wheels_contact(self, state: GameState, player: int) -> bool:
            # TODO: Implement wheel contact detection
            return False

    class Reward(RewardFunction):
        def reset(self, initial_state: GameState):
            self.flip_reset_achieved = False
            self.last_touch_time = 0
            
        def get_reward(self, player: int, state: GameState, previous_action: np.ndarray) -> float:
            reward = 0
            
            # Reward for getting close to ball
            car_pos = state.players[player].car_data.position
            ball_dist = np.linalg.norm(state.ball.position - car_pos)
            reward -= ball_dist * 0.1
            
            # Check for flip reset
            if self._check_flip_reset(state, player):
                if not self.flip_reset_achieved:
                    reward += 20.0
                    self.flip_reset_achieved = True
                    
            # Additional reward for follow-up touch
            if state.last_touch == player:
                touch_time = state.game_info.seconds_elapsed
                if touch_time > self.last_touch_time:
                    if self.flip_reset_achieved:
                        reward += 30.0  # Bonus for flip reset goal
                    else:
                        reward += 5.0
                    self.last_touch_time = touch_time
                    
            return reward
            
        def _check_flip_reset(self, state: GameState, player: int) -> bool:
            # TODO: Implement flip reset detection
            return False

    class Terminal(TerminalCondition):
        def reset(self, initial_state: GameState):
            self.timer = 0
            
        def is_terminal(self, current_state: GameState) -> bool:
            self.timer += 1
            return (
                current_state.ball.position[2] < 100 or  # Ball too low
                self.timer >= 450  # 15 seconds
            )

    class State(StateSetter):
        def reset(self, state_wrapper: GameState):
            # Set up high aerial scenario
            state_wrapper.ball.position = np.array([0, 0, 1500])
            state_wrapper.ball.linear_velocity = np.array([0, 0, -200])
            
            # Place car in position to attempt flip reset
            state_wrapper.players[0].car_data.position = np.array([0, -1000, 1000])
            state_wrapper.players[0].car_data.linear_velocity = np.array([0, 500, 300])
            state_wrapper.players[0].boost_amount = 100