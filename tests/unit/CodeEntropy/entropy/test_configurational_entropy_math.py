import numpy as np
import pytest

from CodeEntropy.entropy.configurational import ConformationalEntropy


def test_conformational_entropy_empty_returns_zero():
    ce = ConformationalEntropy()
    assert ce.conformational_entropy_calculation([]) == 0.0


def test_conformational_entropy_single_state_returns_zero():
    ce = ConformationalEntropy()
    assert ce.conformational_entropy_calculation([0, 0, 0]) == 0.0


def test_conformational_entropy_known_distribution_matches_expected():
    ce = ConformationalEntropy()
    states = np.array([0, 0, 1, 1, 1, 2])

    probs = np.array([2 / 6, 3 / 6, 1 / 6], dtype=float)
    expected = -ce._GAS_CONST * float(np.sum(probs * np.log(probs)))

    got = ce.conformational_entropy_calculation(states)
    assert got == pytest.approx(expected)
