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
