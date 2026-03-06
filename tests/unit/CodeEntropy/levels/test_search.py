import numpy as np
import pytest

from CodeEntropy.levels.search import Search

# some dummy atom positions
a = np.array([0, 0, 1])
b = np.array([0, 1, 0])
c = np.array([1, 0, 0])
d = np.array([0, 1, 1])
e = np.array([0, 11, 11])
dimensions = np.array([10, 10, 10])


def test_get_angle():
    search = Search()
    result1 = search.get_angle(a, b, c, dimensions)
    result2 = search.get_angle(a, b, d, dimensions)

    assert result1 == 0.5
    assert result2 == pytest.approx(0.7071067811865477)


def test_angle_boundary_conditions():
    search = Search()

    result = search.get_angle(a, b, e, dimensions)

    assert result == pytest.approx(0.7071067811865477)


def test_distance():
    search = Search()
    distance1 = search.get_distance(a, b, dimensions)
    distance2 = search.get_distance(a, d, dimensions)
    distance3 = search.get_distance(c, d, dimensions)

    assert distance1 == pytest.approx(1.4142135623730951)
    assert distance2 == 1.0
    assert distance3 == pytest.approx(1.7320508075688772)


def test_distance_boundary_conditions():
    search = Search()

    distance4 = search.get_distance(c, e, dimensions)

    assert distance4 == pytest.approx(1.7320508075688772)
