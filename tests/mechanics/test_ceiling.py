from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import CeilingShotMechanic, CeilingShotParams
from rlbot_pro.state import BallState, CarState, GameState


def build_ground_state() -> GameState:
    ball = BallState(position=Vector3(0.0, 0.0, 120.0), velocity=Vector3(0.0, 0.0, 0.0))
    car = CarState(
        position=Vector3(-800.0, 0.0, 17.0),
        velocity=Vector3(1100.0, 0.0, 0.0),
        boost=60.0,
        has_jump=True,
        is_demolished=False,
        on_ground=True,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def build_detached_state() -> GameState:
    ball = BallState(position=Vector3(0.0, 0.0, 800.0), velocity=Vector3(0.0, 0.0, 0.0))
    car = CarState(
        position=Vector3(-400.0, 0.0, 1900.0),
        velocity=Vector3(1200.0, 0.0, 200.0),
        boost=40.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def test_ceiling_flip_fires_in_window() -> None:
    params = CeilingShotParams(
        carry_target=Vector3(-500.0, 0.0, 2_000.0),
        detach_height=1_800.0,
        detach_time=0.8,
        flip_window=(0.2, 0.3),
    )
    mechanic = CeilingShotMechanic(params)
    ground_state = build_ground_state()
    mechanic.prep(ground_state)
    mechanic.step(ground_state, dt=0.05)
    detached_state = build_detached_state()
    # Advance until we enter the flip window
    flip_triggered = False
    for _ in range(12):
        controls = mechanic.step(detached_state, dt=0.05)
        if controls.jump:
            flip_triggered = True
            break
    assert flip_triggered
