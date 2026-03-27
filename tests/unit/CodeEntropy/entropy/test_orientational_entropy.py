import pytest

from CodeEntropy.entropy.orientational import OrientationalEntropy

_GAS_CONST: float = 8.3144598484848


def test_orientational_negative_count_raises():
    oe = OrientationalEntropy(_GAS_CONST)
    with pytest.raises(ValueError):
        oe.calculate(-1, 1, False, 1, 1)


def test_orientational_zero_count_contributes_zero():
    oe = OrientationalEntropy(_GAS_CONST)
    assert oe.calculate(0, 1, False, 1, 1) == 0.0


def test_omega_linear():
    oe = OrientationalEntropy(_GAS_CONST)
    omega = oe._omega(6, 2, True, 1, 1)
    assert omega == 3.0


def test_omega_nonlinear():
    oe = OrientationalEntropy(_GAS_CONST)
    omega = oe._omega(6, 2, False, 1, 1)
    assert omega == pytest.approx(13.02482)


def test_omega_no_symmetry():
    oe = OrientationalEntropy(_GAS_CONST)
    omega = oe._omega(6, 0, False, 1, 1)
    assert omega == 1.0


def test_omega_hbond_bias():
    oe = OrientationalEntropy(_GAS_CONST)
    omega = oe._omega(6, 1, False, 0.5, 0.5)
    assert omega == pytest.approx(4.60497)
