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

        tournament_cfg_raw = config.get("tournament", {})
        tournament_cfg = tournament_cfg_raw if isinstance(tournament_cfg_raw, Mapping) else {}
        self.tournament_enabled = bool(tournament_cfg.get("enabled", False))
        self.max_style_risk = float(tournament_cfg.get("max_style_risk", 0.65))
        tournament_pressure = float(tournament_cfg.get("pressure_level", self.pressure_level))
        if self.tournament_enabled:
            self.pressure_level = max(self.pressure_level, tournament_pressure)
            self.safety_bias = max(self.safety_bias, 0.45 + 0.35 * self.pressure_level)
            self.aggression = min(self.aggression, 0.55 - 0.2 * self.pressure_level)
        self.min_style_esv = 0.3 + 0.3 * self.pressure_level

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
            option = self._evaluate_option(gs, kind)
            if option is not None:
                candidates.append(option)
        if not candidates:
            pressure = pressure_factor(gs.ball.pos)
            fallback = Option(
                OptionKind.CLEAR,
                esv=self.safety_bias,
                description="clear",
                metadata={"style_risk": 0.2 + 0.3 * pressure, "pressure": pressure},
            )
            self.state.last_option = fallback
            return fallback
        best = max(candidates, key=lambda option: option.esv)
        if best.esv < self.safety_bias:
            pressure = pressure_factor(gs.ball.pos)
            high_pressure = pressure > 0.5
            fallback_kind = OptionKind.CLEAR if high_pressure else OptionKind.CHALLENGE
            fallback_desc = "safety clear" if fallback_kind is OptionKind.CLEAR else "safety challenge"
            fallback_risk = 0.25 + 0.3 * pressure if fallback_kind is OptionKind.CHALLENGE else 0.2 + 0.4 * pressure
            fallback = Option(
                fallback_kind,
                esv=self.safety_bias,
                description=fallback_desc,
                metadata={"style_risk": clamp(fallback_risk, 0.0, 1.0), "pressure": pressure},
            )
            self.state.last_option = fallback
            return fallback
        self.state.last_option = best
        return best

    def _evaluate_option(self, gs: GameState, kind: OptionKind) -> Option | None:
        esv, metadata = self._expected_value(gs, kind)
        if esv <= 0.0:
            return None
        description = f"{kind.name.lower()} esv={esv:.2f}"
        return Option(kind=kind, esv=esv, description=description, metadata=metadata)

    def _expected_value(self, gs: GameState, kind: OptionKind) -> tuple[float, dict[str, float]]:
        pressure = pressure_factor(gs.ball.pos)
        boost_score = clamp(gs.car.boost / 100.0, 0.0, 1.0)
        if kind is OptionKind.RECOVERY:
            grounded_bonus = 1.0 if not gs.car.on_ground else 0.2
            esv = clamp(grounded_bonus * 0.5 + boost_score * 0.2, 0.0, 1.2)
            metadata = {
                "style_risk": clamp(0.15 * (1.0 - boost_score), 0.0, 0.6),
                "pressure": pressure,
                "boost": boost_score,
            }
            return esv, metadata
        time_score = self._time_score(gs)
        angle_score = self._angle_score(gs)
        mechanic_factor = {
            OptionKind.AERIAL: 0.9 if gs.ball.pos[2] > 400 else 0.3,
            OptionKind.AIR_DRIBBLE: 0.7 if gs.ball.pos[2] < 900 else 0.4,
            OptionKind.CEILING: 0.8 if gs.ball.pos[2] > 600 else 0.2,
            OptionKind.FLIP_RESET: 0.85 if gs.ball.pos[2] > 700 else 0.25,
            OptionKind.DOUBLE_TAP: 0.75 if has_shot_line(gs) else 0.3,
        }.get(kind, 0.2)
        base = time_score * 0.3 + angle_score * 0.3 + boost_score * 0.2 + mechanic_factor * 0.2
        aggression_term = self.aggression * (0.5 + pressure * 0.5)
        adjusted = base + aggression_term * 0.2 - self.safety_bias * 0.1
        if not gs.car.on_ground:
            adjusted -= 0.3
        style_risk = self._style_risk(gs, kind, pressure, boost_score)
        if self.tournament_enabled:
            penalty = max(0.0, style_risk - self.max_style_risk)
            adjusted -= penalty * 0.5
            if kind is not OptionKind.RECOVERY:
                adjusted -= clamp(self.min_style_esv - base, 0.0, 0.4)
        esv = clamp(adjusted, 0.0, 1.5)
        metadata = {
            "style_risk": clamp(style_risk, 0.0, 1.0),
            "pressure": pressure,
            "time": time_score,
            "angle": angle_score,
            "boost": boost_score,
        }
        return esv, metadata

    def _time_score(self, gs: GameState) -> float:
        t = time_to_ball_touch(gs)
        if math.isinf(t):
            return 0.0
        return clamp(1.0 - t / 3.0, 0.0, 1.0)

    def _angle_score(self, gs: GameState) -> float:
        if not has_shot_line(gs):
            return 0.2
        return forward_alignment(gs)

    def _style_risk(
        self,
        gs: GameState,
        kind: OptionKind,
        pressure: float,
        boost_score: float,
    ) -> float:
        ball_height = clamp((gs.ball.pos[2] - 100.0) / 1500.0, 0.0, 1.0)
        ball_speed = math.sqrt(sum(component * component for component in gs.ball.vel))
        speed_factor = clamp(ball_speed / 2500.0, 0.0, 1.0)
        base = 0.25 + 0.4 * ball_height + 0.2 * speed_factor
        if kind is OptionKind.AIR_DRIBBLE:
            base += 0.1 * (1.0 - boost_score)
        elif kind in {OptionKind.CEILING, OptionKind.FLIP_RESET}:
            base += 0.2
        elif kind is OptionKind.DOUBLE_TAP:
            base += 0.15 * pressure
        elif kind is OptionKind.AERIAL:
            base += 0.05
        elif kind is OptionKind.RECOVERY:
            base = 0.15 * (1.0 - boost_score)
        elif kind is OptionKind.CLEAR:
            base = 0.2 * pressure
        elif kind is OptionKind.CHALLENGE:
            base = 0.25 + 0.2 * pressure
        return clamp(base, 0.0, 1.0)


__all__ = ["PlannerSelector", "PlannerState"]
