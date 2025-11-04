from dataclasses import replace

from rlbot_pro.math3d import Vector3
from rlbot_pro.sensing import is_ball_heading_to_goal, time_to_ball_touch
from rlbot_pro.sim import build_dummy_state
from rlbot_pro.state import BallState, CarState, GameState


def test_time_to_ball_touch_decreases_with_proximity() -> None:
    base_state = build_dummy_state()
    close_car = replace(base_state.car, position=Vector3(0.0, 0.0, 1100.0))
    closer_state = replace(base_state, car=close_car)
    assert time_to_ball_touch(closer_state) < time_to_ball_touch(base_state)


def test_is_ball_heading_to_goal_alignment() -> None:
    state = GameState(
        ball=BallState(
            position=Vector3(0.0, 0.0, 100.0),
            velocity=Vector3(0.0, 1.0, 0.0),
        ),
        car=CarState(
            position=Vector3(0.0, -1000.0, 17.0),
            velocity=Vector3(0.0, 0.0, 0.0),
            boost=0.0,
            has_jump=True,
            is_demolished=False,
            on_ground=True,
        ),
        time_remaining=300.0,
    )
    assert is_ball_heading_to_goal(state, Vector3(0.0, 1.0, 0.0))
    assert not is_ball_heading_to_goal(state, Vector3(0.0, -1.0, 0.0))
