from dataclasses import replace

from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics.aerial import AerialParams, aerial_step
from rlbot_pro.mechanics.air_dribble import AirDribbleParams, air_dribble_step
from rlbot_pro.mechanics.ceiling import CeilingShotParams, ceiling_shot_step
from rlbot_pro.mechanics.flip_reset import FlipResetParams, flip_reset_step
from rlbot_pro.sim import build_dummy_state


def test_aerial_step_uses_boost_when_target_above() -> None:
    state = build_dummy_state()
    params = AerialParams(target=Vector3(0.0, 0.0, 2000.0), arrival_time=1.0)
    controls = aerial_step(state, params)
    assert controls.boost


def test_air_dribble_step_requests_jump_when_airborne() -> None:
    state = build_dummy_state()
    params = AirDribbleParams(
        carry_offset=Vector3(0.0, 0.0, 120.0),
        target_velocity=1500.0,
    )
    controls = air_dribble_step(state, params)
    assert controls.jump


def test_flip_reset_step_triggers_jump_when_aligned() -> None:
    state = build_dummy_state()
    params = FlipResetParams(
        target_surface_normal=Vector3(0.0, 0.0, 1.0),
        commit_time=0.0,
    )
    controls = flip_reset_step(state, params)
    assert controls.jump


def test_ceiling_shot_step_releases_when_ready() -> None:
    state = build_dummy_state()
    grounded_car = replace(
        state.car,
        position=Vector3(state.car.position.x, state.car.position.y, 17.0),
        on_ground=True,
    )
    grounded_state = replace(state, car=grounded_car)
    params = CeilingShotParams(
        drop_point=Vector3(0.0, 0.0, 2000.0),
        release_time=0.0,
    )
    controls = ceiling_shot_step(grounded_state, params)
    assert controls.jump
