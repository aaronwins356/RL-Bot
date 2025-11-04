from __future__ import annotations

from rlbot_pro.mechanics import aerial
from rlbot_pro.state import BallState, CarState, GameState


def make_state(boost: float = 50.0) -> GameState:
    car = CarState(
        pos=(0.0, -1000.0, 100.0),
        vel=(0.0, 1200.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        forward=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        boost=boost,
        has_flip=True,
        on_ground=False,
        time=0.0,
    )
    ball = BallState(pos=(0.0, 0.0, 800.0), vel=(0.0, 0.0, 0.0))
    return GameState(ball=ball, car=car, dt=1 / 60)


def test_aerial_deterministic_controls() -> None:
    gs = make_state()
    ctrl1 = aerial.step(gs)
    ctrl2 = aerial.step(gs)
    assert ctrl1 == ctrl2
    assert ctrl1.boost


def test_aerial_invalid_when_low_boost() -> None:
    gs = make_state(boost=2.0)
    assert aerial.is_invalid(gs)


def test_aerial_completion_detection() -> None:
    gs = make_state()
    car = CarState(
        pos=(gs.ball.pos[0], gs.ball.pos[1], gs.ball.pos[2] + 10),
        vel=gs.car.vel,
        ang_vel=gs.car.ang_vel,
        forward=gs.car.forward,
        up=gs.car.up,
        boost=gs.car.boost,
        has_flip=gs.car.has_flip,
        on_ground=False,
        time=gs.car.time,
    )
    close_state = GameState(ball=gs.ball, car=car, dt=gs.dt)
    assert aerial.is_complete(close_state, target=gs.ball.pos)
