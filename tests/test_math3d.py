from __future__ import annotations

import math

import pytest

from rlbot_pro.math3d import angle_between, dot, magnitude, normalize, vec_add, vec_scale, vec_sub


def test_vec_add_and_sub() -> None:
    assert vec_add((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert vec_sub((5, 7, 9), (4, 5, 6)) == (1, 2, 3)


def test_normalize_and_magnitude() -> None:
    vec = (3.0, 4.0, 0.0)
    assert magnitude(vec) == pytest.approx(5.0)
    assert normalize(vec) == pytest.approx((0.6, 0.8, 0.0))
    assert normalize((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_angle_between() -> None:
    angle = angle_between((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert math.isclose(angle, math.pi / 2, rel_tol=1e-6)


def test_dot_and_scale() -> None:
    vec = vec_scale((1.0, 2.0, 3.0), 2.0)
    assert vec == (2.0, 4.0, 6.0)
    assert dot(vec, (1.0, 0.0, 0.0)) == 2.0
