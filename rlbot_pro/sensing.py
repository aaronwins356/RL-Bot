import numpy as np
from typing import Tuple

from rlbot_pro.math3d import vec3, normalize, dot, angle_between
from rlbot_pro.state import GameState, CarState, BallState, Vector
from rlbot_pro.physics_helpers import GRAVITY, CAR_MAX_SPEED


def get_car_to_ball_vector(car: CarState, ball: BallState) -> np.ndarray:
    """Returns the vector from the car to the ball."""
    return np.array(ball.pos) - np.array(car.pos)


def get_car_to_target_vector(car: CarState, target: np.ndarray) -> np.ndarray:
    """Returns the vector from the car to a target position."""
    return target - np.array(car.pos)


def get_ball_to_target_vector(ball: BallState, target: np.ndarray) -> np.ndarray:
    """Returns the vector from the ball to a target position."""
    return target - np.array(ball.pos)


def get_car_forward_vector(car: CarState) -> np.ndarray:
    """Returns the car's forward vector as a numpy array."""
    return normalize(np.array(car.forward))


def get_car_right_vector(car: CarState) -> np.ndarray:
    """Returns the car's right vector as a numpy array."""
    return normalize(np.cross(np.array(car.up), np.array(car.forward))) * -1.0


def get_car_up_vector(car: CarState) -> np.ndarray:
    """Returns the car's up vector as a numpy array."""
    return normalize(np.array(car.up))


def get_car_local_coords(car: CarState, target: np.ndarray) -> np.ndarray:
    """
    Transforms a global target position into the car's local coordinate system.
    X: forward/backward, Y: left/right, Z: up/down
    """
    car_pos = np.array(car.pos)
    car_forward = get_car_forward_vector(car)
    car_up = get_car_up_vector(car)
    car_right = get_car_right_vector(car)

    offset = target - car_pos
    local_x = dot(offset, car_forward)
    local_y = dot(offset, car_right)
    local_z = dot(offset, car_up)
    return vec3(local_x, local_y, local_z)


def get_time_to_ball(car: CarState, ball: BallState, max_time: float = 6.0) -> float:
    """
    Estimates the time it will take for the car to reach the ball.
    Simplified linear prediction.
    """
    car_pos = np.array(car.pos)
    car_vel = np.array(car.vel)
    ball_pos = np.array(ball.pos)
    ball_vel = np.array(ball.vel)

    dist = np.linalg.norm(ball_pos - car_pos)
    relative_vel = np.linalg.norm(car_vel - ball_vel)

    if relative_vel > 50:  # If there's significant relative speed
        time_to_reach = dist / relative_vel
    else:  # If relative speed is low, assume max car speed
        time_to_reach = dist / CAR_MAX_SPEED

    return min(time_to_reach, max_time)


def get_angle_to_net(car: CarState, team: int) -> float:
    """
    Calculates the angle from the car's forward vector to the opponent's net.
    Team 0 (blue) attacks +Y, Team 1 (orange) attacks -Y.
    """
    car_forward = get_car_forward_vector(car)
    opponent_goal_pos = vec3(0, 5120 * (1 if team == 0 else -1), 0)
    car_to_goal = normalize(opponent_goal_pos - np.array(car.pos))
    return angle_between(car_forward, car_to_goal)


def get_urgency(car: CarState, ball: BallState) -> float:
    """
    Calculates an urgency score based on ball proximity and threat.
    Higher urgency means the ball is closer and/or more dangerous.
    """
    car_pos = np.array(car.pos)
    ball_pos = np.array(ball.pos)
    ball_vel = np.array(ball.vel)

    dist_to_ball = np.linalg.norm(ball_pos - car_pos)
    time_to_ball = get_time_to_ball(car, ball)

    # Simple threat assessment: ball moving towards our net
    our_goal_y = -5120 if car.pos[1] > 0 else 5120  # Approximate our goal line
    ball_to_our_goal = vec3(0, our_goal_y, 0) - ball_pos
    threat_score = clamp(dot(normalize(ball_vel), normalize(ball_to_our_goal)), 0.0, 1.0)

    # Urgency increases with proximity and threat
    urgency = (1.0 - clamp(dist_to_ball / 6000, 0.0, 1.0)) + threat_score * 0.5
    return clamp(urgency, 0.0, 1.0)


def predict_ball_pos_at_time(ball: BallState, time: float) -> np.ndarray:
    """
    Predicts the ball's position at a future time, considering gravity.
    """
    ball_pos = np.array(ball.pos)
    ball_vel = np.array(ball.vel)
    return ball_pos + ball_vel * time + 0.5 * GRAVITY * time**2


def is_car_on_wall(car: CarState) -> bool:
    """Checks if the car is on a wall (not ground or ceiling)."""
    # A car is on a wall if its up vector is mostly horizontal
    return abs(dot(get_car_up_vector(car), vec3(0, 0, 1))) < 0.5 and car.on_ground


def is_car_on_ceiling(car: CarState) -> bool:
    """Checks if the car is on the ceiling."""
    # A car is on the ceiling if its up vector is mostly downwards
    return dot(get_car_up_vector(car), vec3(0, 0, 1)) < -0.5 and car.on_ground


def is_car_on_ground(car: CarState) -> bool:
    """Checks if the car is on the ground."""
    return car.on_ground and dot(get_car_up_vector(car), vec3(0, 0, 1)) > 0.5
