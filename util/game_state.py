import numpy as np
from typing import List

from rlbot.utils.structures.game_data_struct import GameTickPacket, FieldInfoPacket, PlayerInfo

from .physics_object import PhysicsObject
from .player_data import PlayerData


class GameState:
    def __init__(self, game_info: FieldInfoPacket):
        self.blue_score = 0
        self.orange_score = 0
        self.players: List[PlayerData] = []
        self._on_ground_ticks = np.zeros(64)
        self.last_touch = None

        self.ball: PhysicsObject = PhysicsObject()
        self.inverted_ball: PhysicsObject = PhysicsObject()

        # List of "booleans" (1 or 0)
        self.boost_pads: np.ndarray = np.zeros(game_info.num_boosts, dtype=np.float32)
        self.inverted_boost_pads: np.ndarray = np.zeros_like(self.boost_pads, dtype=np.float32)

    def decode(self, packet: GameTickPacket, ticks_elapsed=1):
        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score

        for i in range(packet.num_boost):
            self.boost_pads[i] = packet.game_boosts[i].is_active
        self.inverted_boost_pads[:] = self.boost_pads[::-1]

        self.ball.decode_ball_data(packet.game_ball.physics)
        self.inverted_ball.invert(self.ball)

        self.players = []
        for i in range(packet.num_cars):
            player = self._decode_player(packet.game_cars[i], i, ticks_elapsed)
            self.players.append(player)

            if player.ball_touched:
                self.last_touch = player.car_id

    def _decode_player(self, player_info: PlayerInfo, index: int, ticks_elapsed: int) -> PlayerData:
        player_data = PlayerData()

        player_data.car_data.decode_car_data(player_info.physics)
        player_data.inverted_car_data.invert(player_data.car_data)

        has_contact = player_info.has_wheel_contact
        altitude = player_data.car_data.position[2]

        if has_contact:
            self._on_ground_ticks[index] = 0
        else:
            self._on_ground_ticks[index] += ticks_elapsed

        recently_on_ground = self._on_ground_ticks[index] <= 6

        player_data.car_id = index
        player_data.team_num = player_info.team
        player_data.is_demoed = player_info.is_demolished
        player_data.on_ground = (has_contact and altitude < 50) or (recently_on_ground and altitude < 120)
        player_data.is_on_wall = has_contact and not player_data.on_ground and altitude >= 50
        player_data.ball_touched = False
        player_data.jumped = player_info.jumped
        player_data.double_jumped = player_info.double_jumped
        player_data.has_flip = not player_info.double_jumped
        player_data.has_jump = player_data.on_ground or not player_info.jumped
        player_data.boost_amount = player_info.boost / 100

        return player_data
