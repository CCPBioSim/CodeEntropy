import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.entropy.configurational import ConformationalEntropy


def test_validate_assignment_config_step_must_be_positive():
    ce = ConformationalEntropy()
    with pytest.raises(ValueError):
        ce.assign_conformation(
            data_container=SimpleNamespace(trajectory=list(range(5))),
            dihedral=MagicMock(value=lambda: 10.0),
            number_frames=5,
            bin_width=30,
            start=0,
            end=5,
            step=0,
        )


def test_validate_assignment_config_bin_width_out_of_range():
    ce = ConformationalEntropy()
    with pytest.raises(ValueError):
        ce.assign_conformation(
            data_container=SimpleNamespace(trajectory=list(range(5))),
            dihedral=MagicMock(value=lambda: 10.0),
            number_frames=5,
            bin_width=0,
            start=0,
            end=5,
            step=1,
        )


def test_validate_assignment_config_warns_when_bin_width_not_dividing_360(caplog):
    ce = ConformationalEntropy()
    caplog.set_level(logging.WARNING)

    data_container = SimpleNamespace(trajectory=list(range(5)))
    dihedral = MagicMock()
    dihedral.value.return_value = 10.0

    ce.assign_conformation(
        data_container=data_container,
        dihedral=dihedral,
        number_frames=5,
        bin_width=7,
        start=0,
        end=5,
        step=1,
    )

    assert any("does not evenly divide 360" in r.message for r in caplog.records)


def test_collect_dihedral_angles_normalizes_negative_values():
    ce = ConformationalEntropy()

    traj_slice = list(range(3))
    dihedral = MagicMock()
    dihedral.value.side_effect = [-10.0, 0.0, 10.0]

    phi = ce._collect_dihedral_angles(traj_slice, dihedral)

    assert phi[0] == pytest.approx(350.0)


def test_to_1d_array_returns_none_for_non_iterable_state_input():
    ce = ConformationalEntropy()
    # int is not iterable -> list(states) raises TypeError -> returns None
    assert ce._to_1d_array(123) is None


def test_find_histogram_peaks_skips_zero_population_bins():
    ce = ConformationalEntropy()

    phi = np.zeros(50, dtype=float)

    peaks = ce._find_histogram_peaks(phi, bin_width=30)

    assert peaks.size >= 1


def test_to_1d_array_returns_none_when_states_is_none():
    ce = ConformationalEntropy()
    assert ce._to_1d_array(None) is None
