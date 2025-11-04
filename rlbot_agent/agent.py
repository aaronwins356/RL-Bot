from __future__ import annotations

import logging
import logging.config
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlbot.agents.base_agent import BaseAgent
from rlbot.agents.base_agent import SimpleControllerState as RLBotSimpleControllerState

from rlbot_pro.adapters.rlbot_adapter import controls_to_simple_controller, packet_to_gamestate
from rlbot_pro.control import Controls
from rlbot_pro.planning.options import OptionKind
from rlbot_pro.policy.agent import ProStyleAgent

try:  # numpy is optional at runtime, guard for lightweight installs
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is optional
    np = None  # type: ignore[assignment]


STYLE_KINDS: tuple[OptionKind, ...] = (
    OptionKind.AERIAL,
    OptionKind.AIR_DRIBBLE,
    OptionKind.CEILING,
    OptionKind.FLIP_RESET,
    OptionKind.DOUBLE_TAP,
)


@dataclass(slots=True)
class KickoffState:
    variant: str | None = None
    frame: int = 0
    active: bool = False

    def reset(self) -> None:
        self.variant = None
        self.frame = 0
        self.active = False


class ProTournamentAgent(BaseAgent):
    def initialize_agent(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self._root = root
        self._settings = self._load_settings(root / "config" / "settings.yaml")
        self._configure_logging(root / "config" / "logging.yaml")
        self.logger = logging.getLogger("rlbot_agent")
        tournament_seed = int(self._settings.get("tournament_seed", 0) or 0)
        if tournament_seed:
            random.seed(tournament_seed)
            if np is not None:
                np.random.seed(tournament_seed % (2**32 - 1))
        self._kickoff_state = KickoffState()
        tournament_cfg = self._settings.get("tournament", {})
        if not isinstance(tournament_cfg, dict):
            tournament_cfg = {}
        self._tournament_enabled = bool(tournament_cfg.get("enabled", False))
        self._max_style_risk = float(tournament_cfg.get("max_style_risk", 0.65))
        self._kickoff_pref = str(tournament_cfg.get("kickoff_variant", "auto")).lower()
        self._tournament_min_esv = 0.35 + 0.25 * float(tournament_cfg.get("pressure_level", 0.5))
        self._option_cache: dict[str, OptionKind] = {"last_safe_kind": OptionKind.CLEAR}
        self._fail_log_cooldown = 0
        self._fail_log_interval = 90
        self._agent = ProStyleAgent(self._settings)
        bot_cfg = self._settings.get("bot", {})
        bot_name = bot_cfg.get("name") if isinstance(bot_cfg, dict) else None
        if isinstance(bot_name, str):
            self.name = bot_name
        developer = bot_cfg.get("developer") if isinstance(bot_cfg, dict) else None
        developer_tag = f" by {developer}" if isinstance(developer, str) else ""
        self.logger.info(
            "Initialized %s%s (team=%s, index=%s) tournament_mode=%s",
            self.name,
            developer_tag,
            self.team,
            self.index,
            self._tournament_enabled,
        )

    def get_output(self, packet) -> RLBotSimpleControllerState:  # type: ignore[override]
        try:
            if self._is_kickoff(packet):
                controls = self._kickoff_controls(packet)
            else:
                self._kickoff_state.reset()
                gamestate = packet_to_gamestate(packet, car_index=self.index)
                controls = self._agent.step(gamestate)
                controls = self._guard_tournament(gamestate, controls)
        except Exception as exc:  # pragma: no cover - defensive path
            controls = self._handle_exception(exc, packet)
        return self._to_rlbot_controller(controls)

    # --- configuration helpers -------------------------------------------------
    def _load_settings(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            message = f"settings file missing at {path}"
            raise FileNotFoundError(message)
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            message = "settings.yaml must define a mapping"
            raise TypeError(message)
        return data

    def _configure_logging(self, path: Path) -> None:
        if not path.exists():
            logging.basicConfig(level=logging.INFO)
            return
        config_data = yaml.safe_load(path.read_text())
        if isinstance(config_data, dict):
            logging.config.dictConfig(config_data)
        else:
            logging.basicConfig(level=logging.INFO)

    # --- kickoff handling ------------------------------------------------------
    def _is_kickoff(self, packet) -> bool:
        game_info = getattr(packet, "game_info", None)
        pause_flag = bool(getattr(game_info, "is_kickoff_pause", False))
        ball = getattr(packet, "game_ball", None)
        physics = getattr(ball, "physics", None)
        location = getattr(physics, "location", None)
        if location is None:
            return pause_flag
        return pause_flag or (abs(getattr(location, "x", 0.0)) < 120 and abs(getattr(location, "y", 0.0)) < 120)

    def _kickoff_controls(self, packet) -> Controls:
        car = packet.game_cars[self.index]
        if not self._kickoff_state.active:
            self._kickoff_state.active = True
            self._kickoff_state.variant = self._choose_kickoff_variant(car)
            self._kickoff_state.frame = 0
        variant = self._kickoff_state.variant or "straight"
        controls = self._run_kickoff_variant(packet, car, variant)
        self._kickoff_state.frame += 1
        return controls

    def _choose_kickoff_variant(self, car) -> str:
        pref = self._kickoff_pref
        if pref not in {"auto", "straight", "diagonal"}:
            pref = "auto"
        physics = getattr(car, "physics", None)
        location = getattr(physics, "location", None)
        car_x = float(getattr(location, "x", 0.0))
        if pref == "straight":
            return "straight"
        if pref == "diagonal":
            return "diagonal_right" if car_x >= 0 else "diagonal_left"
        if abs(car_x) < 300:
            return "straight"
        return "diagonal_right" if car_x >= 0 else "diagonal_left"

    def _run_kickoff_variant(self, packet, car, variant: str) -> Controls:
        ball = packet.game_ball
        car_phys = car.physics
        location = car_phys.location
        velocity = car_phys.velocity
        rotation = car_phys.rotation
        frame = self._kickoff_state.frame
        steer_target = self._steer_to_target(rotation.yaw, location, ball.physics.location)
        throttle = 1.0
        boost = frame < 40
        yaw = steer_target
        steer = steer_target
        pitch = 0.0
        jump = False
        handbrake = False
        if variant == "diagonal_left":
            steer = clamp_value(steer_target + 0.35, -1.0, 1.0)
            yaw = steer
            handbrake = frame < 4
        elif variant == "diagonal_right":
            steer = clamp_value(steer_target - 0.35, -1.0, 1.0)
            yaw = steer
            handbrake = frame < 4
        else:  # straight
            steer = clamp_value(steer_target * 0.7, -0.6, 0.6)
            yaw = steer
        if frame < 8:
            pass
        elif frame < 11:
            jump = True
        elif frame < 14:
            jump = False
        elif frame < 18:
            jump = True
            pitch = -1.0
            yaw = clamp_value(yaw * 0.5, -0.6, 0.6)
        else:
            boost = boost and frame < 60
            pitch = -0.15
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if frame > 20 and speed < 700:
            handbrake = False
            boost = True
        if frame > 60 and speed < 1300:
            return self._boost_pad_fallback(location)
        return Controls(
            throttle=throttle,
            steer=steer,
            pitch=pitch,
            yaw=yaw,
            roll=0.0,
            boost=boost,
            jump=jump,
            handbrake=handbrake,
        )

    def _boost_pad_fallback(self, location) -> Controls:
        steer_bias = clamp_value(float(location.x) / 1800.0, -0.6, 0.6)
        return Controls(
            throttle=1.0,
            steer=-steer_bias,
            pitch=0.0,
            yaw=-steer_bias,
            roll=0.0,
            boost=False,
            jump=False,
            handbrake=False,
        )

    def _steer_to_target(self, yaw: float, location, target) -> float:
        to_target = math.atan2(target.y - location.y, target.x - location.x)
        angle_diff = _normalize_angle(to_target - yaw)
        return clamp_value(angle_diff * 1.5, -1.0, 1.0)

    # --- tournament guard ------------------------------------------------------
    def _guard_tournament(self, gamestate, controls: Controls) -> Controls:
        option = self._agent.selector.state.last_option
        if not self._tournament_enabled or option is None:
            return controls
        metadata = option.metadata or {}
        style_risk = float(metadata.get("style_risk", 0.5))
        pressure = float(metadata.get("pressure", 0.5))
        if option.kind in STYLE_KINDS and (option.esv < self._tournament_min_esv or style_risk > self._max_style_risk):
            safe_kind = self._option_cache.get("last_safe_kind", OptionKind.CLEAR)
            fallback = self._safe_controls(gamestate, safe_kind, pressure)
            self._option_cache["last_safe_kind"] = safe_kind
            self.logger.debug(
                "Tournament guard fallback from %s esv=%.2f risk=%.2f to %s",
                option.kind.name,
                option.esv,
                style_risk,
                safe_kind.name,
            )
            return fallback
        if option.kind in (OptionKind.CLEAR, OptionKind.CHALLENGE, OptionKind.RECOVERY):
            self._option_cache["last_safe_kind"] = option.kind
        elif pressure > 0.6:
            self._option_cache["last_safe_kind"] = OptionKind.CLEAR
        else:
            self._option_cache["last_safe_kind"] = OptionKind.CHALLENGE
        return controls

    def _safe_controls(self, gamestate, kind: OptionKind, pressure: float) -> Controls:
        if kind is OptionKind.CHALLENGE:
            steer = 0.35 if gamestate.ball.pos[0] > gamestate.car.pos[0] else -0.35
            return Controls(throttle=1.0, steer=steer, pitch=0.0, yaw=steer, roll=0.0, boost=pressure < 0.6)
        if kind is OptionKind.RECOVERY:
            return Controls(throttle=0.7, steer=0.0, pitch=-0.3, yaw=0.0, roll=0.0, boost=False, jump=False)
        return Controls(throttle=1.0, steer=0.0, pitch=0.0, yaw=0.0, roll=0.0, boost=True)

    # --- failsafe --------------------------------------------------------------
    def _handle_exception(self, exc: Exception, packet) -> Controls:
        if self._fail_log_cooldown <= 0:
            self.logger.exception("Failsafe engaged: %s", exc)
            self._fail_log_cooldown = self._fail_log_interval
        else:
            self._fail_log_cooldown -= 1
        return self._failsafe_controls(packet)

    def _failsafe_controls(self, packet) -> Controls:
        car = packet.game_cars[self.index]
        ball = packet.game_ball
        car_loc = car.physics.location
        ball_loc = ball.physics.location
        car_yaw = car.physics.rotation.yaw
        to_ball = math.atan2(ball_loc.y - car_loc.y, ball_loc.x - car_loc.x)
        angle_diff = _normalize_angle(to_ball - car_yaw)
        steer = clamp_value(angle_diff * 0.8, -1.0, 1.0)
        throttle = 0.7 + 0.2 * max(0.0, math.cos(angle_diff))
        return Controls(
            throttle=clamp_value(throttle, -1.0, 1.0),
            steer=steer,
            pitch=0.0,
            yaw=steer,
            roll=0.0,
            boost=False,
            jump=False,
            handbrake=False,
        )

    # --- conversions -----------------------------------------------------------
    def _to_rlbot_controller(self, controls: Controls) -> RLBotSimpleControllerState:
        simple = controls_to_simple_controller(controls)
        controller = RLBotSimpleControllerState()
        controller.throttle = clamp_value(simple.throttle, -1.0, 1.0)
        controller.steer = clamp_value(simple.steer, -1.0, 1.0)
        controller.pitch = clamp_value(simple.pitch, -1.0, 1.0)
        controller.yaw = clamp_value(simple.yaw, -1.0, 1.0)
        controller.roll = clamp_value(simple.roll, -1.0, 1.0)
        controller.boost = bool(simple.boost)
        controller.jump = bool(simple.jump)
        controller.handbrake = bool(simple.handbrake)
        return controller


def clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


__all__ = ["ProTournamentAgent"]
