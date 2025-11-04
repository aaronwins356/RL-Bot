"""Deterministic recovery policies (wavedash, half-flip, upright)."""

from __future__ import annotations

from dataclasses import dataclass

from ..control import Controls
from ..math3d import Vector3, clamp, dot, normalize
from ..state import GameState


class RecoveryType:
    WAVEDASH = "wavedash"
    HALF_FLIP = "half_flip"
    UPRIGHT = "upright"


@dataclass(frozen=True)
class RecoveryParams:
    """Parameters describing the recovery policy."""

    strategy: str
    wavedash_height: float = 60.0
    half_flip_duration: float = 0.6
    upright_gain: float = 1.1


class RecoveryMechanic:
    """Execute deterministic recoveries based on the configured strategy."""

    def __init__(self, params: RecoveryParams) -> None:
        self.params = params
        self._prepared = False
        self._elapsed = 0.0
        self._half_flip_phase = 0
        self._complete = False
        self._invalid = False

    def prep(self, state: GameState) -> None:
        self._prepared = True
        self._elapsed = 0.0
        self._half_flip_phase = 0
        self._complete = False
        self._invalid = state.car.is_demolished

    def _wavedash(self, state: GameState, dt: float) -> Controls:
        jump = False
        pitch = 0.0
        yaw = 0.0
        if state.car.position.z > self.params.wavedash_height and state.car.has_jump:
            jump = True
            pitch = -0.4
        throttle = 1.0
        steer = 0.0
        roll = 0.0
        boost = False
        if state.car.on_ground:
            self._complete = True
        return Controls(throttle, steer, pitch, yaw, roll, boost, jump, False)

    def _half_flip(self, state: GameState, dt: float) -> Controls:
        self._elapsed += dt
        jump = False
        pitch = -1.0 if self._half_flip_phase == 0 else 1.0
        if self._half_flip_phase == 0 and state.car.has_jump:
            jump = True
            self._half_flip_phase = 1
        elif self._half_flip_phase == 1 and self._elapsed >= self.params.half_flip_duration:
            self._half_flip_phase = 2
        throttle = 0.8
        steer = 0.0
        roll = 0.0
        if self._half_flip_phase == 2 and state.car.on_ground:
            self._complete = True
        return Controls(throttle, steer, pitch, 0.0, roll, False, jump, False)

    def _upright(self, state: GameState, dt: float) -> Controls:
        up = normalize(state.car.up)
        alignment = dot(up, Vector3(0.0, 0.0, 1.0))
        pitch = clamp(-up.x * self.params.upright_gain, -1.0, 1.0)
        roll = clamp(up.y * self.params.upright_gain, -1.0, 1.0)
        yaw = 0.0
        throttle = 0.3
        steer = 0.0
        boost = False
        jump = False
        if alignment > 0.95 and state.car.on_ground:
            self._complete = True
        return Controls(throttle, steer, pitch, yaw, roll, boost, jump, False)

    def step(self, state: GameState, dt: float = 1.0 / 60.0) -> Controls:
        if not self._prepared:
            self.prep(state)
        if self._invalid:
            return Controls(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)

        if self.params.strategy == RecoveryType.WAVEDASH:
            controls = self._wavedash(state, dt)
        elif self.params.strategy == RecoveryType.HALF_FLIP:
            controls = self._half_flip(state, dt)
        else:
            controls = self._upright(state, dt)

        if state.car.is_demolished:
            self._invalid = True

        return controls

    def is_complete(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._complete

    def is_invalid(self, state: GameState) -> bool:
        if not self._prepared:
            self.prep(state)
        return self._invalid


__all__ = ["RecoveryType", "RecoveryParams", "RecoveryMechanic"]

