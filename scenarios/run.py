"""CLI harness for running deterministic mechanic scenarios."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Callable

from rlbot_pro.control import Controls
from rlbot_pro.math3d import Vector3
from rlbot_pro.mechanics import (
    AerialMechanic,
    AerialParams,
    AirDribbleMechanic,
    AirDribbleParams,
    CeilingShotMechanic,
    CeilingShotParams,
    DoubleTapMechanic,
    DoubleTapParams,
    FlipResetMechanic,
    FlipResetParams,
    RecoveryMechanic,
    RecoveryParams,
    RecoveryType,
)
from rlbot_pro.physics_helpers import BOOST_CONSUMPTION_PER_SECOND, advance_ball, advance_car
from rlbot_pro.state import BallState, CarState, GameState


MechanicFactory = Callable[[GameState], object]
SpawnFunc = Callable[[random.Random], GameState]


@dataclass(frozen=True)
class ScenarioSpec:
    """Specification for a deterministic mechanic scenario."""

    mechanic_factory: MechanicFactory
    spawn: SpawnFunc
    max_steps: int = 240
    dt: float = 1.0 / 60.0


def _spawn_aerial(rng: random.Random) -> GameState:
    ball = BallState(
        position=Vector3(rng.uniform(-200.0, 200.0), rng.uniform(-400.0, 400.0), rng.uniform(1500.0, 2200.0)),
        velocity=Vector3(rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0), -200.0),
    )
    car = CarState(
        position=Vector3(rng.uniform(-600.0, -200.0), rng.uniform(-300.0, 300.0), rng.uniform(200.0, 400.0)),
        velocity=Vector3(rng.uniform(400.0, 700.0), rng.uniform(-100.0, 100.0), rng.uniform(200.0, 300.0)),
        boost=40.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def _spawn_air_dribble(rng: random.Random) -> GameState:
    ball = BallState(
        position=Vector3(-rng.uniform(100.0, 200.0), rng.uniform(-100.0, 100.0), 1400.0),
        velocity=Vector3(0.0, 0.0, 0.0),
    )
    car = CarState(
        position=Vector3(-300.0, 0.0, 900.0),
        velocity=Vector3(700.0, 0.0, 200.0),
        boost=60.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def _spawn_ceiling(rng: random.Random) -> GameState:
    ball = BallState(
        position=Vector3(0.0, rng.uniform(-200.0, 200.0), 150.0),
        velocity=Vector3(0.0, 300.0, 400.0),
    )
    car = CarState(
        position=Vector3(-800.0, 0.0, 17.0),
        velocity=Vector3(1100.0, 0.0, 0.0),
        boost=70.0,
        has_jump=True,
        is_demolished=False,
        on_ground=True,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def _spawn_flip_reset(rng: random.Random) -> GameState:
    ball = BallState(
        position=Vector3(0.0, rng.uniform(-100.0, 100.0), 1100.0),
        velocity=Vector3(0.0, 0.0, -100.0),
    )
    car = CarState(
        position=Vector3(-200.0, 0.0, 900.0),
        velocity=Vector3(800.0, 0.0, 0.0),
        boost=30.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def _spawn_double_tap(rng: random.Random) -> GameState:
    ball = BallState(
        position=Vector3(rng.uniform(-300.0, 300.0), 4_600.0, 1_400.0),
        velocity=Vector3(0.0, 500.0, -200.0),
    )
    car = CarState(
        position=Vector3(0.0, 2_000.0, 800.0),
        velocity=Vector3(1_100.0, 1_000.0, 200.0),
        boost=60.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


def _spawn_recovery(rng: random.Random) -> GameState:
    del rng
    ball = BallState(position=Vector3(0.0, 0.0, 300.0), velocity=Vector3(0.0, 0.0, 0.0))
    car = CarState(
        position=Vector3(200.0, -200.0, 400.0),
        velocity=Vector3(-200.0, 300.0, -600.0),
        boost=20.0,
        has_jump=True,
        is_demolished=False,
        on_ground=False,
    )
    return GameState(ball=ball, car=car, time_remaining=None)


SCENARIOS: dict[str, ScenarioSpec] = {
    "aerial": ScenarioSpec(
        mechanic_factory=lambda state: AerialMechanic(
            AerialParams(intercept=state.ball.position, arrival_time=1.2)
        ),
        spawn=_spawn_aerial,
    ),
    "air_dribble": ScenarioSpec(
        mechanic_factory=lambda _: AirDribbleMechanic(
            AirDribbleParams(carry_offset=Vector3(0.0, 0.0, 120.0), target_velocity=1_600.0)
        ),
        spawn=_spawn_air_dribble,
    ),
    "ceiling": ScenarioSpec(
        mechanic_factory=lambda _: CeilingShotMechanic(
            CeilingShotParams(
                carry_target=Vector3(-500.0, 0.0, 2_000.0),
                detach_height=1_900.0,
                detach_time=0.9,
                flip_window=(0.2, 0.35),
            )
        ),
        spawn=_spawn_ceiling,
    ),
    "flip_reset": ScenarioSpec(
        mechanic_factory=lambda _: FlipResetMechanic(
            FlipResetParams(target_surface_normal=Vector3(0.0, 0.0, 1.0), commit_time=0.6, max_resets=2)
        ),
        spawn=_spawn_flip_reset,
    ),
    "double_tap": ScenarioSpec(
        mechanic_factory=lambda _: DoubleTapMechanic(
            DoubleTapParams(
                backboard_y=5_120.0,
                restitution=0.65,
                first_touch_speed=1_600.0,
                second_arrival_time=0.8,
            )
        ),
        spawn=_spawn_double_tap,
    ),
    "recovery": ScenarioSpec(
        mechanic_factory=lambda _: RecoveryMechanic(
            RecoveryParams(strategy=RecoveryType.UPRIGHT)
        ),
        spawn=_spawn_recovery,
    ),
}


def _advance_state(state: GameState, controls: Controls, dt: float) -> GameState:
    ball = advance_ball(state.ball, dt)
    car = advance_car(state.car, controls, dt)
    return GameState(ball=ball, car=car, time_remaining=state.time_remaining)


def run_episode(spec: ScenarioSpec, rng: random.Random) -> tuple[bool, float, float]:
    state = spec.spawn(rng)
    mechanic = spec.mechanic_factory(state)
    if hasattr(mechanic, "prep"):
        mechanic.prep(state)
    boost_used = 0.0
    time_elapsed = 0.0
    for _ in range(spec.max_steps):
        controls = mechanic.step(state, spec.dt)
        if controls.boost:
            boost_used += BOOST_CONSUMPTION_PER_SECOND * spec.dt
        if mechanic.is_complete(state):
            return True, time_elapsed, boost_used
        if mechanic.is_invalid(state):
            return False, time_elapsed, boost_used
        state = _advance_state(state, controls, spec.dt)
        time_elapsed += spec.dt
    return False, time_elapsed, boost_used


def run_scenario(name: str, seed: int, episodes: int = 5) -> dict[str, float]:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown mechanic scenario '{name}'")
    spec = SCENARIOS[name]
    rng = random.Random(seed)
    successes = 0
    total_time = 0.0
    total_boost = 0.0
    for _ in range(episodes):
        success, time_elapsed, boost_used = run_episode(spec, rng)
        total_time += time_elapsed
        total_boost += boost_used
        if success:
            successes += 1
    failures = episodes - successes
    metrics = {
        "mechanic": name,
        "episodes": episodes,
        "conversion_rate": successes / episodes if episodes else 0.0,
        "average_completion_time": total_time / max(successes, 1),
        "average_boost_used": total_boost / episodes if episodes else 0.0,
        "failures": failures,
        "seed": seed,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic mechanic scenarios")
    parser.add_argument("--mechanic", required=True, choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    metrics = run_scenario(args.mechanic, args.seed, args.episodes)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    main()

