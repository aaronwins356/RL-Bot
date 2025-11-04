"""Three-dimensional vector utilities for Rocket League bots."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import acos, sqrt


@dataclass(frozen=True)
class Vector3:
    """Simple three-dimensional vector representation."""

    x: float
    y: float
    z: float

    def __iter__(self: Vector3) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __add__(self: Vector3, other: Vector3) -> Vector3:
        return add(self, other)

    def __sub__(self: Vector3, other: Vector3) -> Vector3:
        return subtract(self, other)

    def __mul__(self: Vector3, scalar: float) -> Vector3:
        return scale(self, scalar)

    __rmul__ = __mul__


def add(a: Vector3, b: Vector3) -> Vector3:
    """Return the sum of two vectors."""
    return Vector3(a.x + b.x, a.y + b.y, a.z + b.z)


def subtract(a: Vector3, b: Vector3) -> Vector3:
    """Return the difference of two vectors."""
    return Vector3(a.x - b.x, a.y - b.y, a.z - b.z)


def scale(vec: Vector3, scalar: float) -> Vector3:
    """Scale a vector by a scalar."""
    return Vector3(vec.x * scalar, vec.y * scalar, vec.z * scalar)


def dot(a: Vector3, b: Vector3) -> float:
    """Return the dot product of two vectors."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vector3, b: Vector3) -> Vector3:
    """Return the cross product of two vectors."""
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def magnitude(vec: Vector3) -> float:
    """Return the magnitude of a vector."""
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)


def normalize(vec: Vector3) -> Vector3:
    """Return a unit vector pointing in the same direction as the input."""
    length = magnitude(vec)
    if length <= 1e-6:
        return Vector3(0.0, 0.0, 0.0)
    inv = 1.0 / length
    return Vector3(vec.x * inv, vec.y * inv, vec.z * inv)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a scalar value within [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def distance(a: Vector3, b: Vector3) -> float:
    """Return the Euclidean distance between two points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return sqrt(dx * dx + dy * dy + dz * dz)


def angle_between(a: Vector3, b: Vector3) -> float:
    """Return the angle between two vectors in radians."""
    mag_product = magnitude(a) * magnitude(b)
    if mag_product <= 1e-9:
        return 0.0
    cos_theta = clamp(dot(a, b) / mag_product, -1.0, 1.0)
    return acos(cos_theta)


def lerp(a: Vector3, b: Vector3, alpha: float) -> Vector3:
    """Linearly interpolate between two points."""
    return Vector3(
        a.x + (b.x - a.x) * alpha,
        a.y + (b.y - a.y) * alpha,
        a.z + (b.z - a.z) * alpha,
    )


__all__ = [
    "Vector3",
    "add",
    "subtract",
    "scale",
    "dot",
    "cross",
    "magnitude",
    "normalize",
    "clamp",
    "distance",
    "angle_between",
    "lerp",
]
