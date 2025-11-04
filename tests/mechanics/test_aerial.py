from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import AerialMechanic, AerialParams
from rlbot_pro.state import BallState, CarState, GameState


def build_state(forward: Vector3 | None = None) -> GameState:
    ball = BallState(position=Vector3(0.0, 0.0, 1800.0), velocity=Vector3(0.0, 0.0, -200.0))
    car_forward = forward or Vector3(0.8, 0.1, 0.5)
    car = CarState(
        position=Vector3(-400.0, 0.0, 400.0),
        velocity=Vector3(800.0, 0.0, 200.0),
        boost=60.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
        forward=car_forward,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def test_aerial_boost_requires_alignment() -> None:
    params = AerialParams(intercept=Vector3(0.0, 0.0, 1800.0), arrival_time=1.2, boost_alignment_threshold=0.4)
    aligned_state = build_state()
    mech = AerialMechanic(params)
    mech.prep(aligned_state)
    controls = mech.step(aligned_state)
    assert controls.boost

    misaligned_state = build_state(forward=Vector3(-1.0, 0.0, 0.0))
    mech2 = AerialMechanic(params)
    mech2.prep(misaligned_state)
    controls2 = mech2.step(misaligned_state)
    assert not controls2.boost


def test_aerial_step_is_deterministic() -> None:
    params = AerialParams(intercept=Vector3(0.0, 0.0, 1800.0), arrival_time=1.2)
    state = build_state()
    controls_first = AerialMechanic(params).step(state)
    controls_second = AerialMechanic(params).step(state)
    assert controls_first == controls_second
