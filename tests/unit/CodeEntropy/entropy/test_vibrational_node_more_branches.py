from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.entropy.nodes.vibrational import EntropyPair, VibrationalEntropyNode


@pytest.fixture()
def shared():
    frag = MagicMock()
    frag.residues = [MagicMock(resname="RES")]
    ru = MagicMock()
    ru.atoms.fragments = [frag]

    return {
        "run_manager": MagicMock(),
        "args": SimpleNamespace(temperature=298.0, combined_forcetorque=False),
        "groups": {5: [0]},
        "levels": {0: ["united_atom"]},
        "reduced_universe": ru,
        "force_covariances": {"ua": {}, "res": [], "poly": []},
        "torque_covariances": {"ua": {}, "res": [], "poly": []},
        "n_frames": 5,
        "reporter": MagicMock(),
    }


def test_get_group_id_to_index_builds_from_groups(shared):
    node = VibrationalEntropyNode()
    gid2i = node._get_group_id_to_index(shared)
    assert gid2i == {5: 0}


def test_get_ua_frame_counts_returns_empty_when_missing(shared):
    node = VibrationalEntropyNode()
    assert node._get_ua_frame_counts(shared) == {}


def test_compute_force_torque_entropy_returns_zero_when_missing_matrix(shared):
    node = VibrationalEntropyNode()
    ve = MagicMock()
    pair = node._compute_force_torque_entropy(
        ve=ve, temp=298.0, fmat=None, tmat=np.eye(3), highest=True
    )
    assert pair == EntropyPair(trans=0.0, rot=0.0)


def test_compute_force_torque_entropy_returns_zero_when_filter_removes_all(monkeypatch):
    node = VibrationalEntropyNode()
    ve = MagicMock()

    monkeypatch.setattr(
        node._mat_ops, "filter_zero_rows_columns", lambda a, atol: np.array([])
    )

    pair = node._compute_force_torque_entropy(
        ve=ve, temp=298.0, fmat=np.eye(3), tmat=np.eye(3), highest=True
    )
    assert pair == EntropyPair(trans=0.0, rot=0.0)


def test_compute_ft_entropy_returns_zero_when_none():
    node = VibrationalEntropyNode()
    ve = MagicMock()
    assert node._compute_ft_entropy(ve=ve, temp=298.0, ftmat=None) == EntropyPair(
        trans=0.0, rot=0.0
    )


def test_log_molecule_level_results_ft_labels_branch():
    node = VibrationalEntropyNode()
    reporter = MagicMock()

    node._log_molecule_level_results(
        reporter, 1, "residue", EntropyPair(1.0, 2.0), use_ft_labels=True
    )

    reporter.add_results_data.assert_any_call(
        1, "residue", "FTmat-Transvibrational", 1.0
    )
    reporter.add_results_data.assert_any_call(1, "residue", "FTmat-Rovibrational", 2.0)


def test_get_indexed_matrix_out_of_range_returns_none():
    node = VibrationalEntropyNode()
    assert node._get_indexed_matrix([np.eye(3)], 5) is None


def test_run_unknown_level_raises(shared):
    node = VibrationalEntropyNode()
    shared["levels"] = {0: ["nope"]}

    with pytest.raises(ValueError):
        node.run(shared)
