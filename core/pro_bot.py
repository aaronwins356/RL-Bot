"""Deterministic high-skill Rocket League bot logic used by the GUI."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from telemetry import BallPredictionSlice, BallState, BotTelemetry, CarState, ControlOutput

GRAVITY = np.array([0.0, 0.0, -650.0], dtype=float)  # uu/s^2
MAX_CAR_SPEED = 2300.0
MAX_ACCELERATION = 1000.0
BOOST_ACCELERATION = 991.666
BOOST_CONSUMPTION_PER_S = 33.3
MAX_YAW_RATE = math.radians(130.0)
ARENA_EXTENTS = np.array([4096.0, 5120.0, 2044.0])
GOAL_CENTER_BLUE = np.array([0.0, -5120.0, 640.0])
GOAL_CENTER_ORANGE = np.array([0.0, 5120.0, 640.0])


@dataclass
class TargetDecision:
    """Captures the bot's current target and reasoning."""

    point: np.ndarray
    arrival_time: float
    mechanic: str


class ProBot(threading.Thread):
    """Physics-based Rocket League bot with deterministic behaviour."""

    def __init__(
        self,
        telemetry_callback: Optional[Callable[[BotTelemetry], None]] = None,
        tick_rate: float = 1.0 / 120.0,
    ) -> None:
        super().__init__(daemon=True)
        self.telemetry_callback = telemetry_callback
        self.tick_rate = tick_rate
        self.is_running = False
        self._shutdown = threading.Event()
        self._state_lock = threading.Lock()
        self.simulation_time = 0.0

        self.car = CarState()
        self.ball = BallState()
        self.controls = ControlOutput()
        self._target = TargetDecision(point=np.zeros(3), arrival_time=0.0, mechanic="Idle")
        self.reset_mock_arena()

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start_running(self) -> None:
        """Allow the bot logic loop to execute."""

        self.is_running = True

    def stop_running(self) -> None:
        """Pause the bot logic loop without killing the thread."""

        self.is_running = False

    def shutdown(self) -> None:
        """Stop the thread entirely and wait for termination."""

        self._shutdown.set()
        self.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - thread loop
        while not self._shutdown.is_set():
            if not self.is_running:
                time.sleep(0.05)
                continue
            self.simulation_time += self.tick_rate
            self.step(self.tick_rate)
            time.sleep(self.tick_rate)

    def step(self, dt: float) -> None:
        """Execute one deterministic simulation step."""

        with self._state_lock:
            prediction = self.predict_ball_trajectory(self.ball, horizon=2.0)
            target_decision = self.choose_target_point(prediction)
            controls = self.calculate_car_controls(target_decision.point, dt)
            mechanic_used = self.apply_advanced_mechanics(target_decision, controls, dt)
            self.integrate_car_physics(controls, dt)
            self.integrate_ball_physics(dt)
            self.controls = ControlOutput(**vars(controls))
            self._target = TargetDecision(
                point=target_decision.point.copy(),
                arrival_time=target_decision.arrival_time,
                mechanic=mechanic_used,
            )
            telemetry = self.build_telemetry(prediction, mechanic_used, self.controls)

        if self.telemetry_callback:
            self.telemetry_callback(telemetry)

    # ------------------------------------------------------------------
    # Prediction and targeting
    # ------------------------------------------------------------------
    def predict_ball_trajectory(
        self, ball: BallState, horizon: float = 2.0, dt: float = 1.0 / 60.0
    ) -> List[BallPredictionSlice]:
        """Propagate the ball forward deterministically using simple rigid-body physics."""

        positions: List[BallPredictionSlice] = []
        pos = ball.position.copy()
        vel = ball.velocity.copy()
        t = 0.0
        steps = int(horizon / dt)
        for _ in range(steps):
            vel += GRAVITY * dt
            pos += vel * dt
            pos, vel = self.enforce_ball_bounds(pos, vel)
            t += dt
            positions.append(
                BallPredictionSlice(
                    time=t,
                    position=tuple(float(x) for x in pos),
                    velocity=tuple(float(v) for v in vel),
                )
            )
        return positions

    def enforce_ball_bounds(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply basic arena collision logic to keep the ball in play."""

        for axis in range(3):
            limit = ARENA_EXTENTS[axis]
            if axis < 2:
                if pos[axis] > limit:
                    pos[axis] = limit
                    vel[axis] *= -0.6
                elif pos[axis] < -limit:
                    pos[axis] = -limit
                    vel[axis] *= -0.6
            else:
                if pos[axis] < 0.0:
                    pos[axis] = 0.0
                    vel[axis] *= -0.45
        return pos, vel

    def choose_target_point(self, prediction: Sequence[BallPredictionSlice]) -> TargetDecision:
        """Select an intercept point on the predicted trajectory."""

        best_slice = prediction[0]
        best_error = float("inf")
        desired_mechanic = "Ground Drive"
        for slice_ in prediction:
            intercept = np.array(slice_.position)
            travel_time = self.estimate_travel_time(intercept)
            error = abs(travel_time - slice_.time)
            if error < best_error:
                best_slice = slice_
                best_error = error
                desired_mechanic = self.determine_mechanic(intercept)
        return TargetDecision(point=np.array(best_slice.position), arrival_time=best_slice.time, mechanic=desired_mechanic)

    def estimate_travel_time(self, destination: np.ndarray) -> float:
        """Estimate time required to reach destination with deterministic kinematics."""

        displacement = destination - self.car.position
        horizontal_distance = float(np.linalg.norm(displacement[:2]))
        current_speed = float(np.linalg.norm(self.car.velocity))
        if horizontal_distance < 1e-3:
            return 0.0
        accel_time = max((MAX_CAR_SPEED - current_speed) / MAX_ACCELERATION, 0.0)
        accel_distance = current_speed * accel_time + 0.5 * MAX_ACCELERATION * accel_time * accel_time
        if horizontal_distance <= accel_distance and MAX_ACCELERATION > 0:
            a = 0.5 * MAX_ACCELERATION
            b = current_speed
            c = -horizontal_distance
            discriminant = max(b * b - 4.0 * a * c, 0.0)
            return (-b + math.sqrt(discriminant)) / (2.0 * a)
        cruise_distance = max(horizontal_distance - accel_distance, 0.0)
        cruise_time = cruise_distance / MAX_CAR_SPEED
        return accel_time + cruise_time

    def determine_mechanic(self, intercept: np.ndarray) -> str:
        """Heuristically choose which mechanic best reaches the intercept point."""

        height = intercept[2]
        if self.simulation_time < 2.0 and np.linalg.norm(self.car.position[:2]) < 1000.0:
            return "Kickoff"
        if height > 800.0:
            return "Aerial"
        if height > 140.0 and np.linalg.norm(self.car.velocity) > 1200.0:
            return "Flip Reset"
        if height < 100.0 and np.linalg.norm(self.car.velocity) < 600.0:
            return "Wave Dash"
        if self.is_shadowing_needed(intercept):
            return "Shadow Defense"
        return "Ground Drive"

    def is_shadowing_needed(self, intercept: np.ndarray) -> bool:
        """Return True when the ball is approaching our half and defense is preferable."""

        towards_own_goal = intercept[1] < 0.0
        ball_ahead = intercept[1] < self.car.position[1]
        return towards_own_goal and ball_ahead

    # ------------------------------------------------------------------
    # Control synthesis
    # ------------------------------------------------------------------
    def calculate_car_controls(self, target: np.ndarray, dt: float) -> ControlOutput:
        """Generate a control plan to reach the target point."""

        controls = ControlOutput()
        to_target = target - self.car.position
        distance = float(np.linalg.norm(to_target[:2]))
        desired_yaw = math.atan2(to_target[1], to_target[0]) if distance > 1e-3 else self.car.yaw
        yaw_error = self.angle_difference(desired_yaw, self.car.yaw)
        controls.steer = float(np.clip(yaw_error * 2.5, -1.0, 1.0))
        target_speed = min(MAX_CAR_SPEED, distance / max(dt, 1e-3))
        speed_error = target_speed - float(np.linalg.norm(self.car.velocity))
        controls.throttle = float(np.clip(speed_error / MAX_CAR_SPEED, -1.0, 1.0))
        controls.yaw = controls.steer
        controls.pitch = float(np.clip(-to_target[2] * 0.005, -1.0, 1.0))
        controls.roll = 0.0
        if speed_error > 400.0 and self.car.boost > 0.0:
            controls.boost = True
        return controls

    @staticmethod
    def angle_difference(target: float, current: float) -> float:
        diff = (target - current + math.pi) % (2 * math.pi) - math.pi
        return diff

    # ------------------------------------------------------------------
    # Advanced mechanics (placeholders with deterministic logic)
    # ------------------------------------------------------------------
    def apply_advanced_mechanics(self, decision: TargetDecision, controls: ControlOutput, dt: float) -> str:
        """Trigger specific mechanics when heuristics deem them optimal."""

        if decision.mechanic == "Kickoff" and self.kickoff_strategy(controls):
            return "Kickoff"
        if decision.mechanic == "Aerial" and self.attempt_aerial_strike(decision.point, controls, dt):
            return "Aerial Strike"
        if decision.mechanic == "Flip Reset" and self.attempt_flip_reset(decision.point, controls, dt):
            return "Flip Reset"
        if decision.mechanic == "Wave Dash" and self.execute_wave_dash(controls, dt):
            return "Wave Dash"
        if decision.mechanic == "Shadow Defense" and self.shadow_defense_positioning(decision.point, controls):
            return "Shadow Defense"
        if self.needs_half_flip() and self.execute_half_flip(controls):
            return "Half Flip"
        return "Ground Drive"

    def execute_wave_dash(self, controls: ControlOutput, dt: float) -> bool:
        """Perform a deterministic wave dash when landing from a small jump."""

        if self.car.position[2] < 60.0 and self.car.velocity[2] < -200.0 and not self.car.has_jumped:
            self.car.has_jumped = True
            self.car.last_jump_time = self.simulation_time
            controls.jump = True
            controls.pitch = -0.7
            controls.roll = 0.0
            return True
        if self.car.has_jumped and self.simulation_time - self.car.last_jump_time > 0.15:
            controls.jump = False
            self.car.has_jumped = False
        return False

    def attempt_aerial_strike(self, intercept: np.ndarray, controls: ControlOutput, dt: float) -> bool:
        """Align the car with the ball mid-air for an aerial shot."""

        vertical_difference = intercept[2] - self.car.position[2]
        if vertical_difference < 150.0:
            return False
        direction = intercept - self.car.position
        distance = float(np.linalg.norm(direction))
        if distance < 1.0:
            return False
        direction_norm = direction / distance
        controls.boost = self.car.boost > 0.0
        controls.pitch = float(np.clip(-direction_norm[2] * 1.5, -1.0, 1.0))
        controls.yaw = float(np.clip(direction_norm[1] * 1.2, -1.0, 1.0))
        controls.roll = float(np.clip(-direction_norm[0] * 1.2, -1.0, 1.0))
        controls.jump = True
        self.car.has_jumped = True
        return True

    def attempt_flip_reset(self, intercept: np.ndarray, controls: ControlOutput, dt: float) -> bool:
        """Attempt a flip reset by aligning car underside with the ball."""

        if self.car.position[2] < 200.0 or intercept[2] < 300.0:
            return False
        lateral_alignment = np.dot(self.car.velocity[:2], (intercept - self.car.position)[:2])
        if lateral_alignment <= 0.0:
            return False
        controls.pitch = -0.2
        controls.roll = 0.0
        controls.jump = True
        return True

    def execute_half_flip(self, controls: ControlOutput) -> bool:
        """Execute a half-flip to quickly reverse direction."""

        if np.dot(self.car.velocity[:2], self.forward_vector()[:2]) > -300.0:
            return False
        controls.jump = True
        controls.pitch = 1.0
        controls.roll = 0.0
        controls.yaw = 0.0
        return True

    def needs_half_flip(self) -> bool:
        """Return True when reversing rapidly to change direction."""

        forward_speed = np.dot(self.car.velocity[:2], self.forward_vector()[:2])
        return forward_speed < -800.0

    def shadow_defense_positioning(self, intercept: np.ndarray, controls: ControlOutput) -> bool:
        """Place the car between the ball and own goal for shadow defense."""

        goal = GOAL_CENTER_BLUE
        ball_vec = intercept - goal
        desired_position = goal + 0.6 * ball_vec
        offset = desired_position - self.car.position
        controls.throttle = float(np.clip(np.linalg.norm(offset[:2]) / 1500.0, -1.0, 1.0))
        controls.steer = float(
            np.clip(self.angle_difference(math.atan2(offset[1], offset[0]), self.car.yaw) * 2.0, -1.0, 1.0)
        )
        return True

    def kickoff_strategy(self, controls: ControlOutput) -> bool:
        """Speedflip-like kickoff sequence for the opening seconds."""

        time_since_start = self.simulation_time
        if time_since_start > 3.0:
            return False
        controls.throttle = 1.0
        controls.boost = self.car.boost > 0.0
        controls.steer = 0.0
        controls.yaw = 0.0
        if time_since_start > 0.4 and not self.car.has_jumped:
            controls.jump = True
            self.car.has_jumped = True
            self.car.last_jump_time = self.simulation_time
        if self.car.has_jumped and self.simulation_time - self.car.last_jump_time > 0.1:
            controls.pitch = -1.0
            controls.yaw = 0.2
        return True

    # ------------------------------------------------------------------
    # Physics integration
    # ------------------------------------------------------------------
    def integrate_car_physics(self, controls: ControlOutput, dt: float) -> None:
        """Apply the generated controls to the mock car physics."""

        forward = self.forward_vector()
        acceleration = controls.throttle * MAX_ACCELERATION
        self.car.velocity += forward * acceleration * dt
        if controls.boost and self.car.boost > 0.0:
            self.car.velocity += forward * BOOST_ACCELERATION * dt
            self.car.boost = max(self.car.boost - BOOST_CONSUMPTION_PER_S * dt, 0.0)
        speed = np.linalg.norm(self.car.velocity)
        if speed > MAX_CAR_SPEED:
            self.car.velocity *= MAX_CAR_SPEED / speed
        self.car.position += self.car.velocity * dt
        self.car.yaw += controls.steer * MAX_YAW_RATE * dt
        self.car.pitch += controls.pitch * dt * 2.0
        self.car.roll += controls.roll * dt * 2.0
        self.car.position, self.car.velocity = self.enforce_car_bounds(self.car.position, self.car.velocity)
        if self.car.position[2] <= 0.0:
            self.car.position[2] = 0.0
            self.car.velocity[2] = 0.0
            self.car.has_jumped = False

    def enforce_car_bounds(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Constrain the car to the arena, mirroring bounces off walls."""

        for axis in range(2):
            limit = ARENA_EXTENTS[axis]
            if pos[axis] > limit:
                pos[axis] = limit
                vel[axis] *= -0.5
            elif pos[axis] < -limit:
                pos[axis] = -limit
                vel[axis] *= -0.5
        if pos[2] < 0.0:
            pos[2] = 0.0
            vel[2] = 0.0
        return pos, vel

    def integrate_ball_physics(self, dt: float) -> None:
        """Advance the ball simulation deterministically."""

        self.ball.velocity += GRAVITY * dt
        self.ball.position += self.ball.velocity * dt
        self.ball.position, self.ball.velocity = self.enforce_ball_bounds(self.ball.position, self.ball.velocity)

    def forward_vector(self) -> np.ndarray:
        return np.array(
            [
                math.cos(self.car.yaw) * math.cos(self.car.pitch),
                math.sin(self.car.yaw) * math.cos(self.car.pitch),
                math.sin(self.car.pitch),
            ]
        )

    def reset_mock_arena(self) -> None:
        """Initialise a deterministic mock environment for GUI testing."""

        self.car.position[:] = np.array([0.0, -2400.0, 17.0])
        self.car.velocity[:] = np.zeros(3)
        self.car.yaw = math.pi / 2
        self.car.pitch = 0.0
        self.car.roll = 0.0
        self.car.boost = 100.0
        self.car.has_jumped = False
        self.ball.position[:] = np.array([0.0, 0.0, 93.0])
        self.ball.velocity[:] = np.array([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def build_telemetry(
        self, prediction: Sequence[BallPredictionSlice], mechanic_used: str, controls: ControlOutput
    ) -> BotTelemetry:
        """Compile telemetry for GUI consumption."""

        speed = float(np.linalg.norm(self.car.velocity))
        shot_accuracy = self.compute_shot_accuracy(prediction)
        ball_prediction = list(prediction)
        telemetry = BotTelemetry(
            speed=speed,
            boost=self.car.boost,
            shot_accuracy=shot_accuracy,
            mechanic=mechanic_used,
            position=tuple(float(x) for x in self.car.position),
            target=tuple(float(x) for x in self._target.point),
            ball_prediction=ball_prediction,
            controls=ControlOutput(**vars(controls)),
            extra_metrics={
                "sim_time": self.simulation_time,
                "car_yaw_deg": math.degrees(self.car.yaw),
            },
        )
        return telemetry

    def compute_shot_accuracy(self, prediction: Sequence[BallPredictionSlice]) -> float:
        """Estimate shot accuracy based on intercept aim towards opponent goal."""

        if not prediction:
            return 0.0
        intercept = np.array(prediction[0].position)
        goal_vec = GOAL_CENTER_ORANGE - intercept
        norm_goal = np.linalg.norm(goal_vec)
        if norm_goal < 1e-6:
            return 1.0
        shot_dir = goal_vec / norm_goal
        car_to_ball = intercept - self.car.position
        norm_car_ball = np.linalg.norm(car_to_ball)
        if norm_car_ball < 1e-6:
            return 0.0
        car_to_ball_norm = car_to_ball / norm_car_ball
        alignment = float(np.clip(np.dot(shot_dir[:2], car_to_ball_norm[:2]), -1.0, 1.0))
        return (alignment + 1.0) / 2.0
