import pytest

from CodeEntropy.entropy.orientational import OrientationalEntropy


def test_orientational_negative_count_raises():
    oe = OrientationalEntropy(None, None, None, None, None)
    with pytest.raises(ValueError):
        oe.calculate_orientational(-1, 1, False)


def test_orientational_zero_count_contributes_zero():
    oe = OrientationalEntropy(None, None, None, None, None)
    assert oe.calculate_orientational(0, 1, False) == 0.0


def test_omega_linear():
    oe = OrientationalEntropy(None, None, None, None, None)
    omega = oe._omega(6, 2, True)
    assert omega == 3.0


def test_omega_no_symmetry():
    oe = OrientationalEntropy(None, None, None, None, None)
    omega = oe._omega(6, 0, False)
    assert omega == 1.0
