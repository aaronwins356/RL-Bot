from dataclasses import replace

from rlbot_pro.math3d import Vector3
from rlbot_pro.planning.options import OptionType, evaluate, execute
from rlbot_pro.policy.agent import ProStyleAgent
from rlbot_pro.sim import build_dummy_state


def test_evaluate_selects_aerial_for_high_ball() -> None:
    state = build_dummy_state()
    aerial_ball = replace(state.ball, position=Vector3(0.0, 0.0, 1800.0))
    aerial_state = replace(state, ball=aerial_ball)
    assert evaluate(aerial_state) is OptionType.AERIAL


def test_execute_returns_controls() -> None:
    state = build_dummy_state()
    option = execute(state, OptionType.GROUND_DRIVE)
    assert option.option_type is OptionType.GROUND_DRIVE
    assert -1.0 <= option.controls.throttle <= 1.0


def test_agent_step_is_deterministic() -> None:
    state = build_dummy_state()
    agent = ProStyleAgent()
    controls_a = agent.step(state)
    controls_b = agent.step(state)
    assert controls_a == controls_b
