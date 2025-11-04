from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import AirDribbleMechanic, AirDribbleParams
from rlbot_pro.state import BallState, CarState, GameState


def build_state(on_ground: bool) -> GameState:
    ball = BallState(position=Vector3(0.0, 0.0, 1400.0), velocity=Vector3(0.0, 0.0, 0.0))
    car = CarState(
        position=Vector3(0.0, 0.0, 200.0 if on_ground else 900.0),
        velocity=Vector3(500.0, 0.0, 100.0),
        boost=30.0,
        has_jump=True,
        is_demolished=False,
        on_ground=on_ground,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def test_air_dribble_requests_jump_from_ground() -> None:
    state = build_state(on_ground=True)
    params = AirDribbleParams(carry_offset=Vector3(0.0, 0.0, 120.0), target_velocity=1600.0)
    mech = AirDribbleMechanic(params)
    mech.prep(state)
    controls = mech.step(state)
    assert controls.jump


def test_air_dribble_is_deterministic() -> None:
    state = build_state(on_ground=False)
    params = AirDribbleParams(carry_offset=Vector3(0.0, 0.0, 120.0), target_velocity=1400.0)
    first = AirDribbleMechanic(params).step(state)
    second = AirDribbleMechanic(params).step(state)
    assert first == second
