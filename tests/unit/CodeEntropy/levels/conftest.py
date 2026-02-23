from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def args():
    # minimal args object used by nodes
    return SimpleNamespace(grouping="each")


@pytest.fixture
def reduced_universe():
    """
    Minimal Universe-like object:
      - .atoms.fragments exists and is list-like
    """
    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.fragments = []
    return u


@pytest.fixture
def universe_with_fragments():
    """
    Universe with 3 fragments.
    Each fragment can be customized by tests.
    """
    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.fragments = [MagicMock(), MagicMock(), MagicMock()]
    return u


@pytest.fixture
def simple_ts_list():
    # list supports slicing directly: lst[start:end:step]
    return [SimpleNamespace(frame=i) for i in range(10)]


@pytest.fixture
def axes_manager_identity():
    """
    AxesCalculator-like adapter used by ForceTorqueCalculator for displacements.
    Returns positions-center (no PBC).
    """
    mgr = MagicMock()

    def _get_vector(center, positions, box):
        center = np.asarray(center, dtype=float).reshape(3)
        positions = np.asarray(positions, dtype=float)
        return positions - center

    mgr.get_vector.side_effect = _get_vector
    return mgr
