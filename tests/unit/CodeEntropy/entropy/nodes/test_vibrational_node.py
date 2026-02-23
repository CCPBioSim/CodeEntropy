from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.entropy.nodes.vibrational import EntropyPair, VibrationalEntropyNode


@pytest.fixture()
def shared_data_base():
    frag = MagicMock()
    frag.residues = [MagicMock(resname="RES")]
    reduced_universe = MagicMock()
    reduced_universe.atoms.fragments = [frag]

    return {
        "run_manager": MagicMock(),
        "args": SimpleNamespace(temperature=298.0, combined_forcetorque=False),
        "groups": {0: [0]},
        "levels": {0: ["united_atom"]},
        "reduced_universe": reduced_universe,
        "force_covariances": {"ua": {}, "res": [], "poly": []},
        "torque_covariances": {"ua": {}, "res": [], "poly": []},
        "n_frames": 5,
        "reporter": MagicMock(),
    }


@pytest.fixture()
def shared_groups():
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


def test_united_atom_branch_logs_and_stores(shared_data_base, monkeypatch):
    node = VibrationalEntropyNode()

    monkeypatch.setattr(
        node,
        "_compute_united_atom_entropy",
        MagicMock(return_value=EntropyPair(trans=1.0, rot=2.0)),
    )
    monkeypatch.setattr(node, "_log_molecule_level_results", MagicMock())

    out = node.run(shared_data_base)

    assert out["vibrational_entropy"][0]["united_atom"]["trans"] == 1.0
    assert out["vibrational_entropy"][0]["united_atom"]["rot"] == 2.0


def test_residue_branch_noncombined(shared_data_base, monkeypatch):
    node = VibrationalEntropyNode()

    shared_data_base["levels"] = {0: ["residue"]}
    shared_data_base["group_id_to_index"] = {0: 0}
    shared_data_base["force_covariances"]["res"] = [np.eye(3)]
    shared_data_base["torque_covariances"]["res"] = [np.eye(3)]

    monkeypatch.setattr(
        node,
        "_compute_force_torque_entropy",
        MagicMock(return_value=EntropyPair(trans=3.0, rot=4.0)),
    )
    monkeypatch.setattr(node, "_log_molecule_level_results", MagicMock())

    out = node.run(shared_data_base)

    assert out["vibrational_entropy"][0]["residue"]["trans"] == 3.0
    assert out["vibrational_entropy"][0]["residue"]["rot"] == 4.0


def test_polymer_branch_combined_ft_at_highest(shared_data_base, monkeypatch):
    node = VibrationalEntropyNode()

    shared_data_base["args"].combined_forcetorque = True
    shared_data_base["levels"] = {0: ["polymer"]}
    shared_data_base["group_id_to_index"] = {0: 0}
    shared_data_base["forcetorque_covariances"] = {"poly": [np.eye(6)]}

    monkeypatch.setattr(
        node,
        "_compute_ft_entropy",
        MagicMock(return_value=EntropyPair(trans=5.0, rot=6.0)),
    )
    monkeypatch.setattr(node, "_log_molecule_level_results", MagicMock())

    out = node.run(shared_data_base)

    assert out["vibrational_entropy"][0]["polymer"]["trans"] == 5.0
    assert out["vibrational_entropy"][0]["polymer"]["rot"] == 6.0


def test_get_indexed_matrix_typeerror_returns_none():
    node = VibrationalEntropyNode()
    assert node._get_indexed_matrix(mats=123, index=0) is None


def test_get_group_id_to_index_uses_cached_mapping(shared_data):
    node = VibrationalEntropyNode()
    shared_data["group_id_to_index"] = {7: 0}
    assert node._get_group_id_to_index(shared_data) == {7: 0}


def test_get_group_id_to_index_falls_back_to_enumeration(shared_data):
    node = VibrationalEntropyNode()
    shared_data.pop("group_id_to_index", None)
    shared_data["groups"] = {5: [0], 9: [1]}
    assert node._get_group_id_to_index(shared_data) == {5: 0, 9: 1}


def test_run_raises_on_unknown_level(shared_data, monkeypatch):
    node = VibrationalEntropyNode()

    shared_data["levels"] = {0: ["banana"]}
    shared_data["groups"] = {0: [0]}

    shared_data["force_covariances"] = {"ua": {}, "res": [], "poly": []}
    shared_data["torque_covariances"] = {"ua": {}, "res": [], "poly": []}

    with pytest.raises(ValueError):
        node.run(shared_data)


def test_run_united_atom_branch_stores_results(shared_data, monkeypatch):
    node = VibrationalEntropyNode()

    shared_data["levels"] = {0: ["united_atom"]}
    shared_data["groups"] = {0: [0]}
    shared_data["force_covariances"] = {"ua": {}, "res": [], "poly": []}
    shared_data["torque_covariances"] = {"ua": {}, "res": [], "poly": []}

    fake_pair = MagicMock(trans=1.0, rot=2.0)
    monkeypatch.setattr(
        node, "_compute_united_atom_entropy", MagicMock(return_value=fake_pair)
    )
    monkeypatch.setattr(node, "_log_molecule_level_results", MagicMock())

    out = node.run(shared_data)

    assert "vibrational_entropy" in out
    assert shared_data["vibrational_entropy"][0]["united_atom"]["trans"] == 1.0
    assert shared_data["vibrational_entropy"][0]["united_atom"]["rot"] == 2.0


def test_unknown_level_raises(shared_data):
    node = VibrationalEntropyNode()

    shared_data["levels"] = {0: ["invalid"]}
    shared_data["groups"] = {0: [0]}
    shared_data["force_covariances"] = {"ua": {}, "res": [], "poly": []}
    shared_data["torque_covariances"] = {"ua": {}, "res": [], "poly": []}

    with pytest.raises(ValueError):
        node.run(shared_data)


def test_polymer_branch_executes(shared_data, monkeypatch):
    node = VibrationalEntropyNode()

    shared_data["levels"] = {0: ["polymer"]}
    shared_data["groups"] = {0: [0]}

    shared_data["force_covariances"] = {"ua": {}, "res": [], "poly": [MagicMock()]}
    shared_data["torque_covariances"] = {"ua": {}, "res": [], "poly": [MagicMock()]}

    shared_data["reduced_universe"].atoms.fragments = [MagicMock(residues=[])]

    monkeypatch.setattr(
        node,
        "_compute_force_torque_entropy",
        MagicMock(return_value=EntropyPair(trans=1.0, rot=1.0)),
    )
    monkeypatch.setattr(node, "_log_molecule_level_results", MagicMock())

    out = node.run(shared_data)

    assert "vibrational_entropy" in out
    assert out["vibrational_entropy"][0]["polymer"]["trans"] == 1.0
    assert out["vibrational_entropy"][0]["polymer"]["rot"] == 1.0


def test_run_skips_empty_mol_ids_group():
    node = VibrationalEntropyNode()

    shared_groups = {
        "run_manager": MagicMock(),
        "args": SimpleNamespace(temperature=298.0, combined_forcetorque=False),
        "groups": {0: []},
        "levels": {0: ["united_atom"]},
        "reduced_universe": MagicMock(atoms=MagicMock(fragments=[])),
        "force_covariances": {"ua": {}, "res": [], "poly": []},
        "torque_covariances": {"ua": {}, "res": [], "poly": []},
        "n_frames": 5,
        "reporter": None,
    }

    out = node.run(shared_groups)
    assert "vibrational_entropy" in out
    assert out["vibrational_entropy"][0] == {}


def test_get_ua_frame_counts_falls_back_to_empty_when_shape_wrong():
    node = VibrationalEntropyNode()
    assert node._get_ua_frame_counts({"frame_counts": "not-a-dict"}) == {}


def test_compute_united_atom_entropy_logs_residue_data_when_reporter_present():
    node = VibrationalEntropyNode()
    ve = MagicMock()

    node._compute_force_torque_entropy = MagicMock(return_value=EntropyPair(1.0, 2.0))

    reporter = MagicMock()
    residues = [SimpleNamespace(resname="A"), SimpleNamespace(resname="B")]

    out = node._compute_united_atom_entropy(
        ve=ve,
        temp=298.0,
        group_id=7,
        residues=residues,
        force_ua={},
        torque_ua={},
        ua_frame_counts={(7, 0): 3, (7, 1): 4},
        reporter=reporter,
        n_frames_default=10,
        highest=True,
    )

    assert out == EntropyPair(trans=2.0, rot=4.0)
    assert reporter.add_residue_data.call_count == 4


def test_compute_force_torque_entropy_success_calls_vibrational_engine():
    node = VibrationalEntropyNode()
    ve = MagicMock()
    ve.vibrational_entropy_calculation.side_effect = [10.0, 20.0]

    out = node._compute_force_torque_entropy(
        ve=ve,
        temp=298.0,
        fmat=np.eye(3),
        tmat=np.eye(3),
        highest=False,
    )

    assert out == EntropyPair(trans=10.0, rot=20.0)
    assert ve.vibrational_entropy_calculation.call_count == 2


def test_compute_ft_entropy_success_calls_vibrational_engine_for_trans_and_rot():
    node = VibrationalEntropyNode()
    ve = MagicMock()
    ve.vibrational_entropy_calculation.side_effect = [1.5, 2.5]

    out = node._compute_ft_entropy(ve=ve, temp=298.0, ftmat=np.eye(6))

    assert out == EntropyPair(trans=1.5, rot=2.5)
    assert ve.vibrational_entropy_calculation.call_count == 2


def test_log_molecule_level_results_returns_when_reporter_none():
    VibrationalEntropyNode._log_molecule_level_results(
        reporter=None,
        group_id=1,
        level="residue",
        pair=EntropyPair(1.0, 2.0),
        use_ft_labels=False,
    )


def test_log_molecule_level_results_writes_trans_and_rot_labels():
    reporter = MagicMock()
    VibrationalEntropyNode._log_molecule_level_results(
        reporter=reporter,
        group_id=1,
        level="residue",
        pair=EntropyPair(3.0, 4.0),
        use_ft_labels=False,
    )

    reporter.add_results_data.assert_any_call(1, "residue", "Transvibrational", 3.0)
    reporter.add_results_data.assert_any_call(1, "residue", "Rovibrational", 4.0)


def test_get_group_id_to_index_builds_from_groups(shared_groups):
    node = VibrationalEntropyNode()
    gid2i = node._get_group_id_to_index(shared_groups)
    assert gid2i == {5: 0}


def test_get_ua_frame_counts_returns_empty_when_missing(shared_groups):
    node = VibrationalEntropyNode()
    assert node._get_ua_frame_counts(shared_groups) == {}


def test_compute_force_torque_entropy_returns_zero_when_missing_matrix(shared_groups):
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


def test_run_unknown_level_raises(shared_groups):
    node = VibrationalEntropyNode()
    shared_groups["levels"] = {0: ["nope"]}

    with pytest.raises(ValueError):
        node.run(shared_groups)


def test_compute_ft_entropy_returns_zeros_when_filtered_ft_matrix_is_empty(monkeypatch):
    node = VibrationalEntropyNode()
    ve = MagicMock()

    monkeypatch.setattr(
        node._mat_ops,
        "filter_zero_rows_columns",
        lambda _arr, atol: np.empty((0, 0), dtype=float),
    )

    out = node._compute_ft_entropy(ve=ve, temp=298.0, ftmat=np.eye(6))

    assert out == EntropyPair(trans=0.0, rot=0.0)
    ve.vibrational_entropy_calculation.assert_not_called()
