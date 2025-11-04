from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from rlbot_pro.control import Controls
from rlbot_pro.mechanics import aerial, air_dribble, ceiling, double_tap, flip_reset, recoveries
from rlbot_pro.planning import PlannerSelector
from rlbot_pro.planning.options import Option, OptionKind
from rlbot_pro.state import GameState
from telemetry.telemetry import TelemetryRecord, TelemetryWriter


@dataclass(slots=True)
class AgentConfig:
    raw: Mapping[str, Any]


class ProStyleAgent:
    def __init__(self, config: Mapping[str, Any], telemetry: TelemetryWriter | None = None):
        self.config = AgentConfig(raw=dict(config))
        self.selector = PlannerSelector(self.config.raw)
        self.telemetry = telemetry
        self._mechanics: dict[
            OptionKind,
            tuple[Callable[[GameState], bool], Callable[[GameState], Controls]],
        ] = {
            OptionKind.AERIAL: (aerial.is_invalid, aerial.step),
            OptionKind.AIR_DRIBBLE: (air_dribble.is_invalid, air_dribble.step),
            OptionKind.CEILING: (ceiling.is_invalid, ceiling.step),
            OptionKind.FLIP_RESET: (flip_reset.is_invalid, flip_reset.step),
            OptionKind.DOUBLE_TAP: (double_tap.is_invalid, double_tap.step),
            OptionKind.RECOVERY: (recoveries.is_invalid, recoveries.step),
        }

    @classmethod
    def from_settings(
        cls,
        path: Path | str = Path("config/settings.yaml"),
        *,
        telemetry: TelemetryWriter | None = None,
    ) -> ProStyleAgent:
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, MutableMapping):
            message = "settings.yaml must define a mapping"
            raise TypeError(message)
        return cls(dict(data), telemetry=telemetry)

    def step(self, gs: GameState) -> Controls:
        option = self.selector.select(gs)
        controls = self._dispatch(option, gs)
        controls = controls.clamped()
        if self.telemetry is not None:
            record = TelemetryRecord.from_state(gs, option, controls)
            self.telemetry.write(record)
        return controls

    def _dispatch(self, option: Option, gs: GameState) -> Controls:
        handlers = self._mechanics.get(option.kind)
        if handlers is None:
            return self._fallback_controls(option, gs)
        is_invalid, step_fn = handlers
        if is_invalid(gs):
            return self._fallback_controls(option, gs)
        return step_fn(gs)

    def _fallback_controls(self, option: Option, gs: GameState) -> Controls:
        if option.kind is OptionKind.CLEAR:
            steer = 0.0
            throttle = 1.0
            return Controls(
                throttle=throttle,
                steer=steer,
                pitch=0.0,
                yaw=steer,
                roll=0.0,
                boost=True,
            )
        if option.kind is OptionKind.CHALLENGE:
            steer = 0.2 if gs.ball.pos[0] > gs.car.pos[0] else -0.2
            return Controls(
                throttle=1.0,
                steer=steer,
                pitch=0.0,
                yaw=steer,
                roll=0.0,
                boost=False,
            )
        return recoveries.step(gs)


__all__ = ["ProStyleAgent"]
