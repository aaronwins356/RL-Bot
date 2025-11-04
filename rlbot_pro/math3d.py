from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from rlbot_pro.state import Vector


def vec(values: Iterable[float]) -> Vector:
    data = tuple(float(v) for v in values)
    if len(data) != 3:
        message = "Vector requires exactly three components"
        raise ValueError(message)
    return data


def vec_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Vector, scale: float) -> Vector:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def magnitude(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def distance(a: Vector, b: Vector) -> float:
    return magnitude(vec_sub(a, b))


def normalize(a: Vector) -> Vector:
    mag = magnitude(a)
    if mag == 0.0:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / mag
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        message = "minimum cannot exceed maximum"
        raise ValueError(message)
    return max(minimum, min(maximum, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def project(a: Vector, onto: Vector) -> Vector:
    onto_unit = normalize(onto)
    scale = dot(a, onto_unit)
    return vec_scale(onto_unit, scale)


def angle_between(a: Vector, b: Vector) -> float:
    na = normalize(a)
    nb = normalize(b)
    cos_theta = clamp(dot(na, nb), -1.0, 1.0)
    return math.acos(cos_theta)


def rotate2d(vec2: Sequence[float], angle: float) -> tuple[float, float]:
    x, y = vec2
    c = math.cos(angle)
    s = math.sin(angle)
    return (x * c - y * s, x * s + y * c)


def flatten(a: Vector) -> tuple[float, float]:
    return (a[0], a[1])


def signed_angle_2d(a: Vector, b: Vector) -> float:
    ax, ay = flatten(normalize(a))
    bx, by = flatten(normalize(b))
    dot_ab = clamp(ax * bx + ay * by, -1.0, 1.0)
    angle = math.acos(dot_ab)
    cross_z = ax * by - ay * bx
    return angle if cross_z >= 0 else -angle


__all__ = [
    "vec",
    "vec_add",
    "vec_sub",
    "vec_scale",
    "dot",
    "cross",
    "magnitude",
    "distance",
    "normalize",
    "clamp",
    "lerp",
    "project",
    "angle_between",
    "rotate2d",
    "flatten",
    "signed_angle_2d",
]
