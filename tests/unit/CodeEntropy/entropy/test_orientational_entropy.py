import pytest

from CodeEntropy.entropy.orientational import OrientationalEntropy


def test_orientational_skips_water_species():
    oe = OrientationalEntropy(None, None, None, None, None)
    res = oe.calculate({"WAT": 10, "LIG": 2})
    assert res.total > 0


def test_orientational_negative_count_raises():
    oe = OrientationalEntropy(None, None, None, None, None)
    with pytest.raises(ValueError):
        oe.calculate({"LIG": -1})


def test_orientational_zero_count_contributes_zero():
    oe = OrientationalEntropy(None, None, None, None, None)
    res = oe.calculate({"LIG": 0})
    assert res.total == 0.0


def test_orientational_skips_water_resname_all_uppercase():
    oe = OrientationalEntropy(None, None, None, None, None)
    res = oe.calculate({"WAT": 10})
    assert res.total == 0.0


def test_orientational_entropy_skips_water_species():
    oe = OrientationalEntropy(None, None, None, None, None)
    res = oe.calculate({"WAT": 10, "Na+": 2})
    assert res.total > 0.0


def test_orientational_calculate_only_water_returns_zero():
    oe = OrientationalEntropy(None, None, None, None, None)
    res = oe.calculate({"WAT": 5})
    assert res.total == 0.0


def test_calculate_skips_water_species_branch():
    oe = OrientationalEntropy(None, None, None, None, None)
    out = oe.calculate({"WAT": 10, "Na+": 2})

    assert out.total > 0.0


def test_entropy_contribution_returns_zero_when_omega_nonpositive(monkeypatch):
    oe = OrientationalEntropy(None, None, None, None, None)

    monkeypatch.setattr(OrientationalEntropy, "_omega", staticmethod(lambda n: 0.0))

    assert oe._entropy_contribution(5) == 0.0
