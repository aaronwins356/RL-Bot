from __future__ import annotations

from typing import Any

from rlbot_pro.planning.options import OptionKind
from rlbot_pro.planning.selector import PlannerSelector
from rlbot_pro.state import BallState, CarState, GameState


def make_state(
    *,
    ball_height: float = 800.0,
    boost: float = 70.0,
    on_ground: bool = True,
    ball_y: float = 0.0,
) -> GameState:
    car = CarState(
        pos=(0.0, -1500.0, 100.0),
        vel=(0.0, 1500.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        forward=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        boost=boost,
        has_flip=True,
        on_ground=on_ground,
        time=0.0,
    )
    ball = BallState(pos=(0.0, ball_y, ball_height), vel=(0.0, 0.0, 0.0))
    return GameState(ball=ball, car=car, dt=1 / 60)


def default_config() -> dict[str, Any]:
    return {
        "aggression": 0.6,
        "mechanics": {
            "aerial": True,
            "air_dribble": True,
            "ceiling": True,
            "flip_reset": True,
            "double_tap": True,
            "recoveries": True,
        },
        "planner": {"pressure_level": 0.5, "safety_bias": 0.3},
    }


def test_selector_prefers_aerial_for_high_ball() -> None:
    selector = PlannerSelector(default_config())
    option = selector.select(make_state(ball_height=900.0))
    assert option.kind is OptionKind.AERIAL


def test_selector_falls_back_when_disabled() -> None:
    config = default_config()
    config["mechanics"]["aerial"] = False
    config["mechanics"]["air_dribble"] = False
    config["mechanics"]["ceiling"] = False
    config["mechanics"]["flip_reset"] = False
    config["mechanics"]["double_tap"] = False
    selector = PlannerSelector(config)
    option = selector.select(make_state(boost=5.0, ball_height=200.0, ball_y=5000.0))
    assert option.kind in {OptionKind.CHALLENGE, OptionKind.CLEAR}


def test_selector_recovers_when_airborne() -> None:
    selector = PlannerSelector(default_config())
    option = selector.select(make_state(ball_height=200.0, on_ground=False))
    assert option.kind in {OptionKind.RECOVERY, OptionKind.AIR_DRIBBLE, OptionKind.AERIAL}
