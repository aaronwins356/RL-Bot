from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlbot_pro.math3d import clamp
from rlbot_pro.planning.options import Option, OptionKind, mechanic_enabled
from rlbot_pro.sensing import forward_alignment, has_shot_line, pressure_factor, time_to_ball_touch
from rlbot_pro.state import GameState


@dataclass(slots=True)
class PlannerState:
    last_option: Option | None = None


class PlannerSelector:
    def __init__(self, config: Mapping[str, Any]):
        self._config: Mapping[str, Any] = dict(config)
        self.state = PlannerState()
        planner_cfg_raw = config.get("planner", {})
        planner_cfg = planner_cfg_raw if isinstance(planner_cfg_raw, Mapping) else {}
        self.pressure_level = float(planner_cfg.get("pressure_level", 0.5))
        self.safety_bias = float(planner_cfg.get("safety_bias", 0.4))
        self.aggression = float(config.get("aggression", 0.5))

    def select(self, gs: GameState) -> Option:
        candidates: list[Option] = []
        for kind in (
            OptionKind.AERIAL,
            OptionKind.AIR_DRIBBLE,
            OptionKind.CEILING,
            OptionKind.FLIP_RESET,
            OptionKind.DOUBLE_TAP,
            OptionKind.RECOVERY,
        ):
            if not mechanic_enabled(self._config, kind):
                continue
            esv = self._expected_value(gs, kind)
            description = f"{kind.name.lower()} esv={esv:.2f}"
            candidates.append(Option(kind=kind, esv=esv, description=description))
        if not candidates:
            fallback = Option(OptionKind.CLEAR, esv=self.safety_bias, description="clear")
            self.state.last_option = fallback
            return fallback
        best = max(candidates, key=lambda option: option.esv)
        if best.esv < self.safety_bias:
            high_pressure = pressure_factor(gs.ball.pos) > 0.5
            fallback_kind = OptionKind.CLEAR if high_pressure else OptionKind.CHALLENGE
            if fallback_kind is OptionKind.CLEAR:
                fallback_desc = "safety clear"
            else:
                fallback_desc = "safety challenge"
            fallback = Option(fallback_kind, esv=self.safety_bias, description=fallback_desc)
            self.state.last_option = fallback
            return fallback
        self.state.last_option = best
        return best

    def _expected_value(self, gs: GameState, kind: OptionKind) -> float:
        if kind is OptionKind.RECOVERY:
            grounded_bonus = 1.0 if not gs.car.on_ground else 0.2
            boost_term = clamp(gs.car.boost / 100.0, 0.0, 1.0)
            return grounded_bonus * 0.5 + boost_term * 0.2
        time_score = self._time_score(gs)
        angle_score = self._angle_score(gs)
        boost_score = clamp(gs.car.boost / 100.0, 0.0, 1.0)
        pressure = pressure_factor(gs.ball.pos)
        mechanic_factor = {
            OptionKind.AERIAL: 0.9 if gs.ball.pos[2] > 400 else 0.3,
            OptionKind.AIR_DRIBBLE: 0.7 if gs.ball.pos[2] < 900 else 0.4,
            OptionKind.CEILING: 0.8 if gs.ball.pos[2] > 600 else 0.2,
            OptionKind.FLIP_RESET: 0.85 if gs.ball.pos[2] > 700 else 0.25,
            OptionKind.DOUBLE_TAP: 0.75 if has_shot_line(gs) else 0.3,
        }.get(kind, 0.2)
        aggression_term = self.aggression * (0.5 + pressure * 0.5)
        base = time_score * 0.3 + angle_score * 0.3 + boost_score * 0.2 + mechanic_factor * 0.2
        adjusted = base + aggression_term * 0.2 - self.safety_bias * 0.1
        if not gs.car.on_ground:
            adjusted -= 0.3
        return clamp(adjusted, 0.0, 1.5)

    def _time_score(self, gs: GameState) -> float:
        t = time_to_ball_touch(gs)
        if math.isinf(t):
            return 0.0
        return clamp(1.0 - t / 3.0, 0.0, 1.0)

    def _angle_score(self, gs: GameState) -> float:
        if not has_shot_line(gs):
            return 0.2
        return forward_alignment(gs)


__all__ = ["PlannerSelector", "PlannerState"]
