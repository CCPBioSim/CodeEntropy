from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.entropy.configurational import ConformationalEntropy


def test_find_histogram_peaks_empty_histogram_returns_empty():
    ce = ConformationalEntropy()
    phi = np.zeros(100, dtype=float)

    peaks = ce._find_histogram_peaks(phi, bin_width=30)

    assert isinstance(peaks, np.ndarray)
    assert peaks.dtype == float


def test_find_histogram_peaks_returns_empty_for_empty_phi():
    ce = ConformationalEntropy()
    phi = np.array([], dtype=float)

    peaks = ce._find_histogram_peaks(phi, bin_width=30)

    assert isinstance(peaks, np.ndarray)
    assert peaks.size == 0


def test_assign_nearest_peaks_with_single_peak_assigns_all_zero():
    ce = ConformationalEntropy()
    phi = np.array([0.0, 10.0, 20.0], dtype=float)
    peak_values = np.array([15.0], dtype=float)

    states = ce._assign_nearest_peaks(phi, peak_values)

    assert np.all(states == 0)


def test_assign_conformation_no_peaks_returns_all_zero():
    ce = ConformationalEntropy()

    data_container = SimpleNamespace(trajectory=[])
    dihedral = MagicMock()

    states = ce.assign_conformation(
        data_container=data_container,
        dihedral=dihedral,
        number_frames=0,
        bin_width=30,
        start=0,
        end=0,
        step=1,
    )

    assert states.size == 0


def test_assign_conformation_fallback_when_peak_finder_returns_empty(monkeypatch):
    ce = ConformationalEntropy()
    data_container = SimpleNamespace(trajectory=list(range(5)))
    dihedral = MagicMock()
    dihedral.value.return_value = 10.0

    monkeypatch.setattr(
        ce, "_find_histogram_peaks", lambda phi, bw: np.array([], dtype=float)
    )

    states = ce.assign_conformation(
        data_container=data_container,
        dihedral=dihedral,
        number_frames=5,
        bin_width=30,
        start=0,
        end=5,
        step=1,
    )
    assert np.all(states == 0)


def test_assign_conformation_detects_multiple_states():
    ce = ConformationalEntropy()

    values = [0.0] * 50 + [180.0] * 50
    data_container = SimpleNamespace(trajectory=list(range(len(values))))
    dihedral = MagicMock()
    dihedral.value.side_effect = values

    states = ce.assign_conformation(
        data_container=data_container,
        dihedral=dihedral,
        number_frames=len(values),
        bin_width=30,
        start=0,
        end=len(values),
        step=1,
    )

    assert len(np.unique(states)) >= 2


def test_conformational_entropy_empty_returns_zero():
    ce = ConformationalEntropy()
    assert ce.conformational_entropy_calculation([], number_frames=10) == 0.0


def test_conformational_entropy_single_state_returns_zero():
    ce = ConformationalEntropy()
    assert ce.conformational_entropy_calculation([0, 0, 0], number_frames=3) == 0.0


def test_conformational_entropy_known_distribution_matches_expected():
    ce = ConformationalEntropy()
    states = np.array([0, 0, 1, 1, 1, 2])

    probs = np.array([2 / 6, 3 / 6, 1 / 6], dtype=float)
    expected = -ce._GAS_CONST * float(np.sum(probs * np.log(probs)))

    got = ce.conformational_entropy_calculation(states, number_frames=6)
    assert got == pytest.approx(expected)
