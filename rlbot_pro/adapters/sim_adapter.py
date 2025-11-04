from __future__ import annotations

from dataclasses import dataclass

from rlbot_pro.control import Controls
from rlbot_pro.physics_helpers import GRAVITY, integrate_constant_accel
from rlbot_pro.state import BallState, CarState, GameState


@dataclass(slots=True)
class SimConfig:
    dt: float = 1.0 / 60.0
    seed: int = 0


class SimWorld:
    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.dt = self.config.dt
        base_ball_vel_y = 600.0 + self.config.seed * 5.0
        self._car_state = CarState(
            pos=(0.0, -1500.0, 17.0),
            vel=(0.0, 1200.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            forward=(0.0, 1.0, 0.0),
            up=(0.0, 0.0, 1.0),
            boost=60.0,
            has_flip=True,
            on_ground=True,
            time=0.0,
        )
        self._ball_state = BallState(pos=(0.0, 0.0, 93.0), vel=(0.0, base_ball_vel_y, 0.0))
        self.tick_count = 0

    def tick(self, controls: Controls | None = None) -> GameState:
        controls = controls.clamped() if controls is not None else Controls()
        car = self._car_state
        dt = self.dt
        forward = car.forward
        accel = controls.throttle * 1200.0
        ax = forward[0] * accel
        ay = forward[1] * accel
        vx = car.vel[0] + ax * dt
        vy = car.vel[1] + ay * dt
        vz = car.vel[2]
        on_ground = car.on_ground
        if controls.jump and car.on_ground:
            vz = 400.0
            on_ground = False
        if not on_ground:
            _, (vx, vy, vz) = integrate_constant_accel((0.0, 0.0, 0.0), (vx, vy, vz), GRAVITY, dt)
        new_pos = (
            car.pos[0] + vx * dt,
            car.pos[1] + vy * dt,
            max(0.0, car.pos[2] + vz * dt),
        )
        if new_pos[2] <= 0.0:
            vz = 0.0
            on_ground = True
        boost = max(0.0, car.boost - (30.0 * dt if controls.boost else 0.0))
        self._car_state = CarState(
            pos=new_pos,
            vel=(vx, vy, vz),
            ang_vel=car.ang_vel,
            forward=car.forward,
            up=car.up,
            boost=boost,
            has_flip=car.has_flip and not controls.jump,
            on_ground=on_ground,
            time=car.time + dt,
        )
        ball_pos, ball_vel = integrate_constant_accel(
            self._ball_state.pos,
            self._ball_state.vel,
            (0.0, 0.0, -325.0),
            dt,
        )
        if ball_pos[2] <= 92.75:
            ball_pos = (ball_pos[0], ball_pos[1], 93.0)
            ball_vel = (ball_vel[0], ball_vel[1], abs(ball_vel[2]) * 0.6)
        self._ball_state = BallState(pos=ball_pos, vel=ball_vel)
        self.tick_count += 1
        return GameState(ball=self._ball_state, car=self._car_state, dt=dt)


__all__ = ["SimWorld", "SimConfig"]
