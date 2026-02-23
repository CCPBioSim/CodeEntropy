from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.entropy.nodes.vibrational import EntropyPair, VibrationalEntropyNode


def test_run_skips_empty_mol_ids_group():
    node = VibrationalEntropyNode()

    shared = {
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

    out = node.run(shared)
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
