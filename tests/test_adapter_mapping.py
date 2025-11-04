from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from rlbot_pro.adapters.rlbot_adapter import (
    PacketLike,
    SimpleControllerState,
    controls_to_simple_controller,
    packet_to_gamestate,
)
from rlbot_pro.control import Controls


def fake_packet() -> object:
    ball = SimpleNamespace(
        physics=SimpleNamespace(
            location=SimpleNamespace(x=0, y=0, z=100),
            velocity=SimpleNamespace(x=0, y=500, z=0),
        )
    )
    car = SimpleNamespace(
        physics=SimpleNamespace(
            location=SimpleNamespace(x=10, y=-1000, z=17),
            velocity=SimpleNamespace(x=0, y=1500, z=0),
            rotation=SimpleNamespace(pitch=0.0, yaw=1.57, roll=0.0),
            angular_velocity=SimpleNamespace(x=0, y=0, z=0),
        ),
        boost=50,
        double_jumped=False,
        has_wheel_contact=True,
    )
    game_info = SimpleNamespace(seconds_elapsed=12.3)
    return SimpleNamespace(game_ball=ball, game_cars=[car], game_info=game_info)


def test_packet_to_gamestate() -> None:
    gs = packet_to_gamestate(cast(PacketLike, fake_packet()))
    assert gs.ball.pos[2] == 100
    assert gs.car.pos[1] == -1000
    assert gs.car.boost == 50


def test_controls_mapping() -> None:
    ctrl = Controls(
        throttle=1.0,
        steer=-0.5,
        pitch=0.1,
        yaw=-0.2,
        roll=0.0,
        boost=True,
        jump=False,
        handbrake=True,
    )
    simple = controls_to_simple_controller(ctrl)
    assert isinstance(simple, SimpleControllerState)
    assert simple.throttle == 1.0
    assert simple.handbrake is True
