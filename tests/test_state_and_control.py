from __future__ import annotations

import pytest

from rlbot_pro.control import Controls
from rlbot_pro.state import BallState, CarState, GameState


def test_state_dataclasses_are_immutable() -> None:
    car = CarState(
        pos=(0, 0, 0),
        vel=(0, 0, 0),
        ang_vel=(0, 0, 0),
        forward=(0, 1, 0),
        up=(0, 0, 1),
        boost=150.0,
        has_flip=True,
        on_ground=True,
        time=0.0,
    )
    with pytest.raises(AttributeError):
        car.pos = (1, 1, 1)  # type: ignore[misc]
    assert car.boost == 100.0
    ball = BallState(pos=(0, 0, 0), vel=(0, 0, 0))
    gs = GameState(ball=ball, car=car, dt=1 / 60)
    assert gs.car is car


def test_game_state_requires_positive_dt() -> None:
    car = CarState(
        pos=(0, 0, 0),
        vel=(0, 0, 0),
        ang_vel=(0, 0, 0),
        forward=(0, 1, 0),
        up=(0, 0, 1),
        boost=0,
        has_flip=False,
        on_ground=True,
        time=0.0,
    )
    ball = BallState(pos=(0, 0, 0), vel=(0, 0, 0))
    with pytest.raises(ValueError):
        GameState(ball=ball, car=car, dt=0)


def test_controls_clamping() -> None:
    ctrl = Controls(
        throttle=2.0,
        steer=-2.0,
        pitch=1.5,
        yaw=-1.5,
        roll=3.0,
        boost=True,
        jump=True,
        handbrake=True,
    )
    clamped = ctrl.clamped()
    assert clamped.throttle == 1.0
    assert clamped.steer == -1.0
    assert clamped.roll == 1.0
