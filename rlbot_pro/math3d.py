"""Three-dimensional vector utilities for Rocket League bots."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass


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
    return math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)


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
]
