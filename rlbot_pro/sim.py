"""Standalone simulation helpers for running the policy without RLBot."""

from __future__ import annotations

from .control import Controls
from .math3d import Vector3
from .policy.agent import ProStyleAgent
from .state import BallState, CarState, GameState


def build_dummy_state() -> GameState:
    """Create a deterministic game state for quick experiments."""
    ball = BallState(
        position=Vector3(0.0, 0.0, 1200.0),
        velocity=Vector3(0.0, 0.0, -200.0),
    )
    car = CarState(
        position=Vector3(-500.0, 0.0, 300.0),
        velocity=Vector3(500.0, 0.0, 200.0),
        boost=50.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=60.0)


def run_dummy_frame() -> Controls:
    """Execute the policy on a deterministic game state and return controls."""
    agent = ProStyleAgent()
    state = build_dummy_state()
    return agent.step(state)


if __name__ == "__main__":
    controls = run_dummy_frame()
    print(controls)
