from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import FlipResetMechanic, FlipResetParams
from rlbot_pro.state import BallState, CarState, GameState


def build_state(up: Vector3) -> GameState:
    ball = BallState(position=Vector3(0.0, 0.0, 1150.0), velocity=Vector3(0.0, 0.0, 0.0))
    car = CarState(
        position=Vector3(0.0, 0.0, 1000.0),
        velocity=Vector3(900.0, 0.0, 0.0),
        boost=20.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
        up=up,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def test_flip_reset_triggers_jump_after_contact() -> None:
    params = FlipResetParams(target_surface_normal=Vector3(0.0, 0.0, 1.0), commit_time=0.0, max_resets=1, flip_cooldown=0.0)
    state = build_state(up=Vector3(0.0, 0.0, -1.0))
    mechanic = FlipResetMechanic(params)
    mechanic.prep(state)
    mechanic.step(state, dt=0.05)
    controls = mechanic.step(state, dt=0.05)
    assert controls.jump


def test_flip_reset_determinism() -> None:
    params = FlipResetParams(target_surface_normal=Vector3(0.0, 0.0, 1.0), commit_time=0.2)
    state = build_state(up=Vector3(0.0, 0.0, 1.0))
    first = FlipResetMechanic(params).step(state)
    second = FlipResetMechanic(params).step(state)
    assert first == second
