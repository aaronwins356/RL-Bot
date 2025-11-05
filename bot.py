from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket, FieldInfoPacket

import numpy as np
from agent import Agent
from obs import CustomObs
from sequences.speedflip import Speedflip
from sequences.wavedash import WaveDash
from sequences.halfflip import HalfFlip
from util.game_state import GameState
from util.ball_prediction import BallPredictor
from util.boost_manager import BoostManager


class WinYour1s(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.obs_builder = CustomObs(cars=2)
        self.action_trans = np.array([-1, -1, -1, -1, -1, 0, 0, 0])
        self.agent = Agent(self.obs_builder.obs_size, action_categoricals=5, action_bernoullis=3)
        self.tick_skip = 4  # REDUCED from 8 to 4 for faster reactions (30 FPS)
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.kickoff_seq = None
        self.wavedash_seq = WaveDash()
        self.halfflip_seq = HalfFlip()
        self.ball_predictor = None  # Will be initialized in initialize_agent
        self.boost_manager = None   # Will be initialized in initialize_agent
        print('WinYour1s Ready - Index:', index)

    def is_hot_reload_enabled(self):
        return True

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        field_info = self.get_field_info()
        self.game_state = GameState(field_info)
        self.update_action = True
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.kickoff_seq = None
        self.wavedash_seq = WaveDash()
        self.halfflip_seq = HalfFlip()
        
        # Initialize SSL-level systems
        self.ball_predictor = BallPredictor(prediction_horizon=4.0, timestep=1/60)
        self.boost_manager = BoostManager(field_info)
        
        print('WinYour1s - SSL systems initialized: Ball Prediction + Boost Management')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)
        
        # Update SSL systems
        if self.boost_manager is not None:
            self.boost_manager.update(self.game_state, delta)

        if packet.game_info.is_kickoff_pause:
            try:
                player = self.game_state.players[self.index]
                teammates = [p for p in self.game_state.players if p.team_num == self.team]
                closest = min(teammates, key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))

                if self.kickoff_seq is None:
                    self.kickoff_seq = Speedflip(player)

                if player == closest and self.kickoff_seq.is_valid(player, self.game_state):
                    self.action = np.asarray(self.kickoff_seq.get_action(player, self.game_state, self.action))
                    self.update_controls(self.action)
                    return self.controls
            except:
                print('WinYour1s - Kickoff sequence failed, falling back to model')
        else:
            self.kickoff_seq = None

        # Check for flashy mechanics (WaveDash)
        player = self.game_state.players[self.index]
        if self.wavedash_seq.is_valid(player, self.game_state) and self.wavedash_seq.is_finished():
            # Trigger wavedash if conditions are met and the sequence is not running
            # For simplicity, we'll trigger it when the bot is near the ball and needs a quick speed boost
            ball_dist = np.linalg.norm(self.game_state.ball.position - player.car_data.position)
            if ball_dist < 1000 and np.linalg.norm(player.car_data.linear_velocity) < 1000:
                self.wavedash_seq.reset()

        # Check for flashy mechanics (HalfFlip)
        if self.halfflip_seq.is_valid(player, self.game_state) and self.halfflip_seq.is_finished():
            # Trigger halfflip if the bot is moving slowly and needs to turn around
            # The condition for moving backward is complex, so we'll simplify the trigger for now.
            # A full implementation would require more complex vector math.
            if np.linalg.norm(player.car_data.linear_velocity) < 500 and player.car_data.position[2] < 50: # Simple trigger: slow and on ground
                self.halfflip_seq.reset()

        if not self.halfflip_seq.is_finished():
            self.action = np.asarray(self.halfflip_seq.get_action(player, self.game_state, self.action))
            self.update_controls(self.action)
            return self.controls

        if not self.wavedash_seq.is_finished():
            self.action = np.asarray(self.wavedash_seq.get_action(player, self.game_state, self.action))
            self.update_controls(self.action)
            return self.controls


        # We calculate the next action as soon as the prev action is sent
        # This gives you tick_skip ticks to do your forward pass
        if self.update_action:
            self.update_action = False

            # This model is 1v1, remove the extra players from the state
            player = self.game_state.players[self.index]

            teammates = [p for p in self.game_state.players if p.team_num == self.team]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            # Sort by distance to ball
            teammates.sort(key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
            opponents.sort(key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))

            # Grab opponent in same "position" as Element relative to it's teammates
            opponent = opponents[min(teammates.index(player), len(opponents) - 1)]

            self.game_state.players = [player, opponent]

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(obs)[0] + self.action_trans

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = 0 if action[5] > 0 else action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
