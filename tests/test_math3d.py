from rlbot_pro.math3d import (
    Vector3,
    add,
    clamp,
    cross,
    dot,
    magnitude,
    normalize,
    scale,
    subtract,
)


def test_vector_operations() -> None:
    a = Vector3(1.0, 2.0, 3.0)
    b = Vector3(-4.0, 5.0, -6.0)
    assert add(a, b) == Vector3(-3.0, 7.0, -3.0)
    assert subtract(a, b) == Vector3(5.0, -3.0, 9.0)
    assert scale(a, 2.0) == Vector3(2.0, 4.0, 6.0)
    assert dot(a, b) == -12.0
    assert cross(a, b) == Vector3(-27.0, -6.0, 13.0)
    assert magnitude(a) == magnitude(Vector3(-1.0, -2.0, -3.0))


def test_normalize_handles_small_vectors() -> None:
    zeroed = normalize(Vector3(1e-9, 0.0, 0.0))
    assert zeroed == Vector3(0.0, 0.0, 0.0)


def test_clamp_bounds() -> None:
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(11.0, 0.0, 10.0) == 10.0
