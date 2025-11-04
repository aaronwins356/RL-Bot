from __future__ import annotations

from rlbot_pro.math3d import Vector3
from rlbot_pro.physics_helpers import CubicSpline, sample_spline


def test_cubic_spline_interpolates_endpoints() -> None:
    times = [0.0, 0.5, 1.0]
    points = [Vector3(0.0, 0.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(2.0, 0.0, 0.0)]
    spline = CubicSpline(times, points)
    samples = sample_spline(spline, [0.0, 0.5, 1.0])
    assert samples[0] == points[0]
    assert samples[-1] == points[-1]
