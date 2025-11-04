from __future__ import annotations

from scenarios.run import run_scenario


def test_scenario_metrics_deterministic() -> None:
    first = run_scenario("aerial", seed=42, episodes=3)
    second = run_scenario("aerial", seed=42, episodes=3)
    assert first == second


def test_scenario_cli_handles_failures() -> None:
    metrics = run_scenario("double_tap", seed=0, episodes=2)
    assert metrics["failures"] >= 0
