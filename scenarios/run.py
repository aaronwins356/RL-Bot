from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from rlbot_pro.adapters.sim_adapter import SimConfig, SimWorld
from rlbot_pro.policy.agent import ProStyleAgent
from telemetry.metrics import TelemetryMetrics
from telemetry.telemetry import TelemetryWriter


def run(mechanic: str, seed: int, ticks: int) -> None:
    settings_path = Path("config/settings.yaml")
    config_raw = yaml.safe_load(settings_path.read_text()) or {}
    if not isinstance(config_raw, dict):
        message = "settings.yaml must contain a mapping"
        raise TypeError(message)
    config: dict[str, Any] = dict(config_raw)
    mechanics_cfg = {
        key: False
        for key in ("aerial", "air_dribble", "ceiling", "flip_reset", "double_tap", "recoveries")
    }
    if mechanic not in mechanics_cfg:
        message = f"Unknown mechanic: {mechanic}"
        raise ValueError(message)
    mechanics_cfg[mechanic] = True
    config["mechanics"] = mechanics_cfg
    telemetry_value = config.get("telemetry_csv", "runs/telemetry.csv")
    telemetry_path = Path(str(telemetry_value))
    world = SimWorld(SimConfig(seed=seed))
    with TelemetryWriter(telemetry_path) as telemetry_writer:
        agent = ProStyleAgent(config, telemetry=telemetry_writer)
        state = world.tick()
        for _ in range(ticks):
            controls = agent.step(state)
            state = world.tick(controls)
    metrics = TelemetryMetrics.from_csv(telemetry_path)
    metrics_path = Path("runs/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RLBot Pro scenarios")
    parser.add_argument("--mechanic", required=True, help="Mechanic to enable for the scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=120)
    args = parser.parse_args(argv)
    run(args.mechanic, args.seed, args.ticks)


if __name__ == "__main__":
    main()
