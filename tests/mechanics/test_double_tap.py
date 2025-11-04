from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import DoubleTapMechanic, DoubleTapParams
from rlbot_pro.state import BallState, CarState, GameState


def build_state(ball_velocity: float) -> GameState:
    ball = BallState(
        position=Vector3(0.0, 4_600.0, 1_200.0),
        velocity=Vector3(0.0, ball_velocity, -200.0),
    )
    car = CarState(
        position=Vector3(0.0, 2_000.0, 800.0),
        velocity=Vector3(900.0, 800.0, 0.0),
        boost=40.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def test_double_tap_invalid_when_ball_moving_away() -> None:
    params = DoubleTapParams(
        backboard_y=5_120.0,
        restitution=0.6,
        first_touch_speed=1_600.0,
        second_arrival_time=0.8,
    )
    state = build_state(ball_velocity=-400.0)
    mechanic = DoubleTapMechanic(params)
    mechanic.prep(state)
    assert mechanic.is_invalid(state)


def test_double_tap_deterministic_controls() -> None:
    params = DoubleTapParams(
        backboard_y=5_120.0,
        restitution=0.6,
        first_touch_speed=1_600.0,
        second_arrival_time=0.8,
    )
    state = build_state(ball_velocity=400.0)
    first = DoubleTapMechanic(params).step(state)
    second = DoubleTapMechanic(params).step(state)
    assert first == second
