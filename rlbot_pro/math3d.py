import numpy as np
from typing import Tuple

Vector = Tuple[float, float, float]


def vec3(x: float, y: float, z: float) -> np.ndarray:
    """Creates a 3D numpy vector."""
    return np.array([x, y, z], dtype=float)


def to_vec3(v: Vector) -> np.ndarray:
    """Converts a tuple Vector to a numpy array."""
    return np.array(v, dtype=float)


def to_tuple(v: np.ndarray) -> Vector:
    """Converts a numpy array to a tuple Vector."""
    return (v[0], v[1], v[2])


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.array([0.0, 0.0, 0.0])


def dot(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the dot product of two vectors."""
    return np.dot(v1, v2)


def cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Computes the cross product of two vectors."""
    return np.cross(v1, v2)


def look_at(target: np.ndarray, current_up: np.ndarray) -> np.ndarray:
    """
    Creates a rotation matrix that makes the 'forward' vector point towards 'target'.
    'current_up' is used to define the 'up' direction.
    Returns a 3x3 rotation matrix.
    """
    forward = normalize(target)
    left = normalize(cross(current_up, forward))
    up = normalize(cross(forward, left))
    return np.array([forward, left, up]).T


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Creates a 3x3 rotation matrix from Euler angles (roll, pitch, yaw).
    Order of application: yaw, pitch, roll.
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Yaw matrix
    R_z = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    # Pitch matrix
    R_y = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    # Roll matrix
    R_x = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    return R_z @ R_y @ R_x


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(value, max_val))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the angle in radians between two vectors."""
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    return np.arccos(clamp(dot(v1_norm, v2_norm), -1.0, 1.0))


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotates a vector around an axis by a given angle using Rodrigues' rotation formula."""
    axis = normalize(axis)
    return v * np.cos(angle) + cross(axis, v) * np.sin(angle) + axis * dot(axis, v) * (1 - np.cos(angle))
