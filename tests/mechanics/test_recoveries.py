from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import RecoveryMechanic, RecoveryParams, RecoveryType
from rlbot_pro.state import BallState, CarState, GameState


def build_state(on_ground: bool) -> GameState:
    ball = BallState(position=Vector3(0.0, 0.0, 300.0), velocity=Vector3(0.0, 0.0, 0.0))
    car = CarState(
        position=Vector3(200.0, -200.0, 30.0 if on_ground else 300.0),
        velocity=Vector3(-200.0, 100.0, -300.0),
        boost=10.0,
        has_jump=True,
        is_demolished=False,
        on_ground=on_ground,
        up=Vector3(0.2, 0.0, 0.5),
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def test_wavedash_triggers_jump_when_airborne() -> None:
    state = build_state(on_ground=False)
    params = RecoveryParams(strategy=RecoveryType.WAVEDASH, wavedash_height=100.0)
    mech = RecoveryMechanic(params)
    mech.prep(state)
    controls = mech.step(state)
    assert controls.jump


def test_upright_completion_when_aligned_on_ground() -> None:
    state = build_state(on_ground=True)
    params = RecoveryParams(strategy=RecoveryType.UPRIGHT)
    mech = RecoveryMechanic(params)
    mech.prep(state)
    controls = mech.step(state)
    assert not mech.is_complete(state)
    grounded_state = GameState(
        ball=state.ball,
        car=CarState(
            position=state.car.position,
            velocity=state.car.velocity,
            boost=state.car.boost,
            has_jump=state.car.has_jump,
            is_demolished=False,
            on_ground=True,
            up=Vector3(0.0, 0.0, 1.0),
        ),
        time_remaining=None,
    )
    mech.step(grounded_state)
    assert mech.is_complete(grounded_state)
