from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rlbot_pro.control import Controls
from rlbot_pro.state import BallState, CarState, GameState, Vector


@dataclass(slots=True)
class SimpleControllerState:
    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    boost: bool = False
    jump: bool = False
    handbrake: bool = False


_RLBotControllerCls: type[Any] | None

try:  # pragma: no cover - import guard
    _rlbot_module = importlib.import_module("rlbot.agents.base_agent")
except ModuleNotFoundError:  # pragma: no cover - executed when rlbot missing
    RLBot_AVAILABLE = False
    _RLBotControllerCls = None
else:
    RLBot_AVAILABLE = True
    controller_cls = getattr(_rlbot_module, "SimpleControllerState", None)
    _RLBotControllerCls = controller_cls if isinstance(controller_cls, type) else None


class VectorLike(Protocol):
    x: float
    y: float
    z: float


class RotationLike(Protocol):
    pitch: float
    yaw: float
    roll: float


class PhysicsLike(Protocol):
    location: VectorLike
    velocity: VectorLike
    rotation: RotationLike
    angular_velocity: VectorLike


class BallLike(Protocol):
    physics: PhysicsLike


class CarLike(Protocol):
    physics: PhysicsLike
    boost: float
    double_jumped: bool
    has_wheel_contact: bool


class GameInfoLike(Protocol):
    seconds_elapsed: float | None


class PacketLike(Protocol):
    game_ball: BallLike
    game_cars: Sequence[CarLike]
    game_info: GameInfoLike


class PacketProvider(Protocol):
    def __call__(self) -> PacketLike: ...


class ControlSender(Protocol):
    def __call__(self, controls: object) -> None: ...


def _vec_from(obj: VectorLike) -> Vector:
    return (float(obj.x), float(obj.y), float(obj.z))


def packet_to_gamestate(packet: PacketLike) -> GameState:
    ball = packet.game_ball
    car = packet.game_cars[0]
    ball_state = BallState(
        pos=_vec_from(ball.physics.location),
        vel=_vec_from(ball.physics.velocity),
    )
    rotation = car.physics.rotation
    forward = (float(rotation.pitch), float(rotation.yaw), float(rotation.roll))
    car_state = CarState(
        pos=_vec_from(car.physics.location),
        vel=_vec_from(car.physics.velocity),
        ang_vel=_vec_from(car.physics.angular_velocity),
        forward=forward,
        up=(0.0, 0.0, 1.0),
        boost=float(car.boost),
        has_flip=bool(car.double_jumped),
        on_ground=bool(car.has_wheel_contact),
        time=float(packet.game_info.seconds_elapsed or 0.0),
    )
    return GameState(ball=ball_state, car=car_state, dt=1.0 / 60.0)


def controls_to_simple_controller(ctrl: Controls) -> SimpleControllerState:
    cmd = ctrl.clamped()
    return SimpleControllerState(
        throttle=cmd.throttle,
        steer=cmd.steer,
        pitch=cmd.pitch,
        yaw=cmd.yaw,
        roll=cmd.roll,
        boost=cmd.boost,
        jump=cmd.jump,
        handbrake=cmd.handbrake,
    )


def _to_rlbot_controller(ctrl: SimpleControllerState) -> object:
    if _RLBotControllerCls is None:
        return ctrl
    controller = _RLBotControllerCls()
    controller.throttle = ctrl.throttle
    controller.steer = ctrl.steer
    controller.pitch = ctrl.pitch
    controller.yaw = ctrl.yaw
    controller.roll = ctrl.roll
    controller.boost = ctrl.boost
    controller.jump = ctrl.jump
    controller.handbrake = ctrl.handbrake
    return controller


def tick_loop(
    agent_step: Callable[[GameState], Controls],
    packet_provider: PacketProvider,
    control_sender: ControlSender,
    tick_rate: float = 1.0 / 60.0,
    max_ticks: int | None = None,
) -> None:
    if not RLBot_AVAILABLE:
        message = "rlbot package is required for tick_loop; install rlbot to continue"
        raise ImportError(message)
    tick = 0
    while max_ticks is None or tick < max_ticks:
        packet = packet_provider()
        gs = packet_to_gamestate(packet)
        controls = agent_step(gs)
        simple = controls_to_simple_controller(controls)
        control_sender(_to_rlbot_controller(simple))
        time.sleep(tick_rate)
        tick += 1


__all__ = [
    "packet_to_gamestate",
    "controls_to_simple_controller",
    "tick_loop",
    "SimpleControllerState",
]
