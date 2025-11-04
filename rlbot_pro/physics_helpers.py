import numpy as np
from typing import Tuple

from rlbot_pro.math3d import vec3, normalize, dot, clamp
from rlbot_pro.state import GameState, BallState, CarState, Vector

# Constants for Rocket League physics
GRAVITY = vec3(0, 0, -650)  # Units per second squared
CAR_MAX_SPEED = 2300  # Units per second
BOOST_ACCELERATION = 991.667  # Units per second squared
CAR_DRAG_COEFF = -0.03  # Simplified drag coefficient


def get_car_forward_vector(car: CarState) -> np.ndarray:
    """Returns the car's forward vector as a numpy array."""
    return normalize(np.array(car.forward))


def get_car_up_vector(car: CarState) -> np.ndarray:
    """Returns the car's up vector as a numpy array."""
    return normalize(np.array(car.up))


def constant_acceleration_kinematics(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    acceleration: np.ndarray,
    time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates position and velocity after 'time' seconds under constant acceleration.
    """
    final_pos = initial_pos + initial_vel * time + 0.5 * acceleration * time**2
    final_vel = initial_vel + acceleration * time
    return final_pos, final_vel


def estimate_ball_bounce(
    ball: BallState,
    car_pos: np.ndarray,
    max_time: float = 6.0,
    steps: int = 60,
) -> Tuple[np.ndarray, float] | None:
    """
    Estimates the ball's next bounce point and time on the ground or wall.
    Simplified for backboard bounces.
    Returns (bounce_pos, bounce_time) or None if no bounce within max_time.
    """
    sim_ball_pos = np.array(ball.pos)
    sim_ball_vel = np.array(ball.vel)
    sim_ball_ang_vel = np.array(ball.ang_vel)
    dt = max_time / steps

    for i in range(steps):
        # Apply gravity
        sim_ball_vel += GRAVITY * dt
        sim_ball_pos += sim_ball_vel * dt

        # Simple ground/wall collision (very basic, assumes flat ground/walls)
        if sim_ball_pos[2] < 93:  # Ground height
            sim_ball_pos[2] = 93
            sim_ball_vel[2] *= -0.7  # Simple bounce, 70% restitution
            if np.linalg.norm(sim_ball_vel) < 100:  # Ball comes to rest
                sim_ball_vel = vec3(0, 0, 0)

        # Check for wall collision (simplified for backboard)
        # Assuming standard field dimensions, backboards are at Y = +/- 5120
        if abs(sim_ball_pos[1]) > 5100:
            sim_ball_pos[1] = np.sign(sim_ball_pos[1]) * 5100
            sim_ball_vel[1] *= -0.7  # Simple bounce
            return sim_ball_pos, i * dt

        # Check if ball is moving away from car or too far
        if np.linalg.norm(sim_ball_pos - car_pos) > 6000:
            return None

    return None


def cubic_spline_point(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    """
    Calculates a point on a cubic Bezier spline.
    t should be between 0 and 1.
    """
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3


def get_car_facing_vector(car: CarState) -> np.ndarray:
    """Returns the car's facing vector (forward) as a numpy array."""
    return np.array(car.forward)


def get_car_right_vector(car: CarState) -> np.ndarray:
    """Returns the car's right vector as a numpy array."""
    # Cross product of up and forward gives left, so negate for right
    return np.cross(np.array(car.up), np.array(car.forward)) * -1.0


def get_car_local_coords(car: CarState, target: np.ndarray) -> np.ndarray:
    """
    Transforms a global target position into the car's local coordinate system.
    X: forward/backward, Y: left/right, Z: up/down
    """
    car_pos = np.array(car.pos)
    car_forward = get_car_facing_vector(car)
    car_up = get_car_up_vector(car)
    car_right = get_car_right_vector(car)

    offset = target - car_pos
    local_x = dot(offset, car_forward)
    local_y = dot(offset, car_right)
    local_z = dot(offset, car_up)
    return vec3(local_x, local_y, local_z)


def get_car_angular_velocity_local(car: CarState) -> np.ndarray:
    """
    Transforms the car's global angular velocity into its local coordinate system.
    """
    car_forward = get_car_facing_vector(car)
    car_up = get_car_up_vector(car)
    car_right = get_car_right_vector(car)
    ang_vel_global = np.array(car.ang_vel)

    local_pitch_rate = dot(ang_vel_global, car_right)
    local_yaw_rate = dot(ang_vel_global, car_up)
    local_roll_rate = dot(ang_vel_global, car_forward)
    return vec3(local_pitch_rate, local_yaw_rate, local_roll_rate)
