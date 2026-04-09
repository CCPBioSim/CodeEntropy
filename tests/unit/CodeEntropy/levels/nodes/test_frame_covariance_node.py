from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from CodeEntropy.levels.nodes.covariance import FrameCovarianceNode


class _BeadGroup:
    def __init__(self, n=1):
        self._n = n

    def __len__(self):
        return self._n

    def center_of_mass(self, unwrap=False):
        return np.array([0.0, 0.0, 0.0], dtype=float)


class _EmptyGroup:
    def __len__(self):
        return 0


def _mk_atomgroup(n=1):
    g = MagicMock()
    g.__len__.return_value = n
    return g


def test_get_shared_missing_raises_keyerror():
    node = FrameCovarianceNode()
    with pytest.raises(KeyError):
        node._get_shared({})


def test_try_get_box_returns_none_on_failure():
    node = FrameCovarianceNode()
    u = MagicMock()
    type(u).dimensions = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
    assert node._try_get_box(u) is None


def test_accumulate_sum_first_sample_copies():
    node = FrameCovarianceNode()
    new = np.eye(2)
    out = node._accumulate_sum(None, new)
    np.testing.assert_allclose(out, new)
    new[0, 0] = 999.0
    assert out[0, 0] != 999.0


def test_accumulate_sum_adds_arrays():
    node = FrameCovarianceNode()
    old = np.array([[2.0, 2.0], [2.0, 2.0]])
    new = np.array([[4.0, 0.0], [0.0, 4.0]])
    out = node._accumulate_sum(old, new)
    np.testing.assert_allclose(out, np.array([[6.0, 2.0], [2.0, 6.0]]))


def test_build_ft_block_rejects_mismatched_lengths():
    node = FrameCovarianceNode()
    with pytest.raises(ValueError):
        node._build_ft_block([np.zeros(3)], [np.zeros(3), np.zeros(3)])


def test_build_ft_block_rejects_empty():
    node = FrameCovarianceNode()
    with pytest.raises(ValueError):
        node._build_ft_block([], [])


def test_build_ft_block_rejects_non_length3_vectors():
    node = FrameCovarianceNode()
    with pytest.raises(ValueError):
        node._build_ft_block([np.zeros(2)], [np.zeros(3)])


def test_build_ft_block_returns_symmetric_block_matrix():
    node = FrameCovarianceNode()

    force_vecs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])]
    torque_vecs = [np.array([0.0, 0.0, 3.0]), np.array([4.0, 0.0, 0.0])]

    M = node._build_ft_block(force_vecs, torque_vecs)
    assert M.shape == (12, 12)

    np.testing.assert_allclose(M, M.T)


def test_process_residue_skips_when_no_beads_key_present():
    node = FrameCovarianceNode()

    shared = {
        "reduced_universe": MagicMock(),
        "groups": {0: [0]},
        "levels": [["residue"]],
        "beads": {},
        "args": MagicMock(
            force_partitioning=1.0, combined_forcetorque=False, customised_axes=False
        ),
        "axes_manager": MagicMock(),
    }
    ctx = {"shared": shared}

    out = node.run(ctx)
    assert out["force"]["res"] == {}
    assert out["torque"]["res"] == {}
    assert "forcetorque" not in out
    assert out["force_counts"]["res"] == {}
    assert out["torque_counts"]["res"] == {}


def test_process_residue_combined_only_when_highest_level():
    node = FrameCovarianceNode()

    u = MagicMock()
    u.atoms = MagicMock()
    frag = MagicMock()
    frag.residues = [MagicMock()]
    u.atoms.fragments = [frag]
    u.atoms.__getitem__.side_effect = lambda idx: _mk_atomgroup(1)
    u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])

    args = MagicMock()
    args.force_partitioning = 1.0
    args.combined_forcetorque = True
    args.customised_axes = True

    axes_manager = MagicMock()
    axes_manager.get_residue_axes.return_value = (
        np.eye(3),
        np.eye(3),
        np.zeros(3),
        np.array([1.0, 1.0, 1.0]),
    )

    shared = {
        "reduced_universe": u,
        "groups": {7: [0]},
        "levels": [["residue"]],
        "beads": {(0, "residue"): [np.array([1, 2, 3])]},
        "args": args,
        "axes_manager": axes_manager,
    }

    with (
        patch.object(
            node._ft, "get_weighted_forces", return_value=np.array([1.0, 0.0, 0.0])
        ),
        patch.object(
            node._ft, "get_weighted_torques", return_value=np.array([0.0, 1.0, 0.0])
        ),
        patch.object(
            node._ft,
            "compute_frame_covariance",
            return_value=(np.eye(3), 2.0 * np.eye(3)),
        ),
    ):
        ctx = {"shared": shared}
        out = node.run(ctx)

    assert "forcetorque" in out
    assert 7 in out["force"]["res"]
    assert 7 in out["torque"]["res"]
    assert 7 in out["forcetorque"]["res"]
    assert out["force_counts"]["res"][7] == 1
    assert out["torque_counts"]["res"][7] == 1
    assert out["forcetorque_counts"]["res"][7] == 1


def test_process_residue_combined_not_added_if_not_highest_level():
    node = FrameCovarianceNode()

    u = MagicMock()
    u.atoms = MagicMock()
    frag = MagicMock()
    frag.residues = [MagicMock()]
    u.atoms.fragments = [frag]
    u.atoms.__getitem__.side_effect = lambda idx: _mk_atomgroup(1)
    u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])

    args = MagicMock(
        force_partitioning=1.0, combined_forcetorque=True, customised_axes=True
    )

    axes_manager = MagicMock()
    axes_manager.get_residue_axes.return_value = (
        np.eye(3),
        np.eye(3),
        np.zeros(3),
        np.ones(3),
    )

    shared = {
        "reduced_universe": u,
        "groups": {7: [0]},
        "levels": [["united_atom", "residue", "polymer"]],
        "beads": {(0, "residue"): [np.array([1, 2, 3])]},
        "args": args,
        "axes_manager": axes_manager,
    }

    with (
        patch.object(
            node._ft, "get_weighted_forces", return_value=np.array([1.0, 0.0, 0.0])
        ),
        patch.object(
            node._ft, "get_weighted_torques", return_value=np.array([0.0, 1.0, 0.0])
        ),
        patch.object(
            node._ft,
            "compute_frame_covariance",
            return_value=(np.eye(3), 2.0 * np.eye(3)),
        ),
    ):
        out = node.run({"shared": shared})

    assert "forcetorque" in out
    assert out["forcetorque"]["res"] == {}
    assert out["force_counts"]["res"][7] == 1
    assert out["torque_counts"]["res"][7] == 1


def test_process_united_atom_returns_when_no_beads_for_level():
    node = FrameCovarianceNode()

    res = MagicMock()
    res.atoms = MagicMock()
    mol = MagicMock()
    mol.residues = [res]

    axes_manager = MagicMock()

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_united_atom(
        u=MagicMock(),
        mol=mol,
        mol_id=0,
        group_id=0,
        beads={},
        axes_manager=axes_manager,
        box=np.array([10.0, 10.0, 10.0], dtype=float),
        force_partitioning=1.0,
        customised_axes=False,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
    )

    assert out_force["ua"] == {}
    assert out_torque["ua"] == {}
    assert out_counts["ua"] == {}
    axes_manager.get_UA_axes.assert_not_called()
    axes_manager.get_vanilla_axes.assert_not_called()


def test_get_residue_axes_vanilla_branch_returns_arrays(monkeypatch):
    node = FrameCovarianceNode()

    monkeypatch.setattr(
        "CodeEntropy.levels.nodes.covariance.make_whole", lambda _ag: None
    )

    mol = MagicMock()
    mol.atoms.principal_axes.return_value = np.eye(3) * 2

    bead = MagicMock()
    bead.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])

    axes_manager = MagicMock()
    axes_manager.get_vanilla_axes.return_value = (np.eye(3), np.array([9.0, 8.0, 7.0]))

    trans, rot, center, moi = node._get_residue_axes(
        mol=mol,
        bead=bead,
        local_res_i=0,
        axes_manager=axes_manager,
        customised_axes=False,
    )

    assert trans.shape == (3, 3)
    assert rot.shape == (3, 3)
    assert center.shape == (3,)
    assert moi.shape == (3,)
    assert np.allclose(trans, np.eye(3) * 2)
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(center, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(moi, np.array([9.0, 8.0, 7.0]))


def test_get_polymer_axes_returns_arrays(monkeypatch):
    node = FrameCovarianceNode()

    monkeypatch.setattr(
        "CodeEntropy.levels.nodes.covariance.make_whole", lambda _ag: None
    )

    mol = MagicMock()
    mol.atoms.principal_axes.return_value = np.eye(3) * 3

    bead = MagicMock()
    bead.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

    axes_manager = MagicMock()
    axes_manager.get_vanilla_axes.return_value = (np.eye(3), np.array([1.0, 1.0, 1.0]))

    trans, rot, center, moi = node._get_polymer_axes(
        mol=mol,
        bead=bead,
        axes_manager=axes_manager,
    )

    assert trans.shape == (3, 3)
    assert rot.shape == (3, 3)
    assert center.shape == (3,)
    assert moi.shape == (3,)
    assert np.allclose(trans, np.eye(3) * 3)
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(moi, np.array([1.0, 1.0, 1.0]))


def test_process_united_atom_updates_outputs_and_counts():
    node = FrameCovarianceNode()

    node._build_ua_vectors = MagicMock(
        return_value=(
            [np.array([1.0, 0.0, 0.0])],
            [np.array([0.0, 1.0, 0.0])],
        )
    )

    F = np.eye(3)
    T = np.eye(3) * 2
    node._ft.compute_frame_covariance = MagicMock(return_value=(F, T))

    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.__getitem__.side_effect = lambda idx: _BeadGroup(1)

    res = MagicMock()
    res.atoms = MagicMock()
    mol = MagicMock()
    mol.residues = [res]

    beads = {(0, "united_atom", 0): [123]}
    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_united_atom(
        u=u,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads=beads,
        axes_manager=MagicMock(),
        box=np.array([10.0, 10.0, 10.0]),
        force_partitioning=1.0,
        customised_axes=False,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
    )

    key = (7, 0)
    assert np.allclose(out_force["ua"][key], F)
    assert np.allclose(out_torque["ua"][key], T)
    assert out_counts["ua"][key] == 1


def test_process_residue_returns_early_when_no_beads():
    node = FrameCovarianceNode()

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_residue(
        u=MagicMock(),
        mol=MagicMock(),
        mol_id=0,
        group_id=0,
        beads={},
        axes_manager=MagicMock(),
        box=np.array([10.0, 10.0, 10.0]),
        customised_axes=False,
        force_partitioning=1.0,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
        out_ft=None,
        out_ft_counts=None,
        combined=False,
    )

    assert out_force["res"] == {}
    assert out_torque["res"] == {}
    assert out_counts["res"] == {}


def test_process_united_atom_skips_when_any_bead_group_is_empty():
    node = FrameCovarianceNode()

    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.__getitem__.side_effect = lambda idx: _EmptyGroup()

    res = MagicMock()
    res.atoms = MagicMock()
    mol = MagicMock()
    mol.residues = [res]

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_united_atom(
        u=u,
        mol=mol,
        mol_id=0,
        group_id=0,
        beads={(0, "united_atom", 0): [1]},
        axes_manager=MagicMock(),
        box=np.array([10.0, 10.0, 10.0]),
        force_partitioning=1.0,
        customised_axes=False,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
    )

    assert out_force["ua"] == {}
    assert out_torque["ua"] == {}
    assert out_counts["ua"] == {}


def test_process_residue_returns_early_when_any_bead_group_is_empty():
    node = FrameCovarianceNode()

    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.__getitem__.side_effect = lambda idx: _EmptyGroup()

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_residue(
        u=u,
        mol=MagicMock(),
        mol_id=0,
        group_id=0,
        beads={(0, "residue"): [1]},
        axes_manager=MagicMock(),
        box=np.array([10.0, 10.0, 10.0]),
        customised_axes=False,
        force_partitioning=1.0,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
        out_ft={},
        out_ft_counts={},
        combined=False,
    )

    assert out_force["res"] == {}
    assert out_torque["res"] == {}
    assert out_counts["res"] == {}


def test_process_polymer_skips_when_any_bead_group_is_empty():
    node = FrameCovarianceNode()

    u = MagicMock()
    u.atoms = MagicMock()
    u.atoms.__getitem__.side_effect = lambda idx: _EmptyGroup()

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_polymer(
        u=u,
        mol=MagicMock(),
        mol_id=0,
        group_id=0,
        beads={(0, "polymer"): [1]},
        axes_manager=MagicMock(),
        box=np.array([10.0, 10.0, 10.0]),
        force_partitioning=1.0,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
        out_ft=None,
        out_ft_counts=None,
        combined=False,
    )

    assert out_force["poly"] == {}
    assert out_torque["poly"] == {}
    assert out_counts["poly"] == {}


def test_process_polymer_happy_path_updates_force_torque_and_optional_ft():
    node = FrameCovarianceNode()

    node._get_polymer_axes = MagicMock(
        return_value=(np.eye(3), np.eye(3), np.zeros(3), np.ones(3))
    )
    node._build_ft_block = MagicMock(return_value=np.eye(6))
    node._ft.get_weighted_forces = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
    node._ft.get_weighted_torques = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))
    node._ft.compute_frame_covariance = MagicMock(
        return_value=(np.eye(3), 2.0 * np.eye(3))
    )

    u = MagicMock()
    u.atoms = MagicMock()
    bead = _BeadGroup(1)
    u.atoms.__getitem__.side_effect = lambda idx: bead

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_counts = {"ua": {}, "res": {}, "poly": {}}
    out_ft = {"ua": {}, "res": {}, "poly": {}}
    out_ft_counts = {"ua": {}, "res": {}, "poly": {}}

    node._process_polymer(
        u=u,
        mol=MagicMock(),
        mol_id=0,
        group_id=7,
        beads={(0, "polymer"): [1]},
        axes_manager=MagicMock(),
        box=np.array([10.0, 10.0, 10.0]),
        force_partitioning=1.0,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_counts=out_counts,
        out_ft=out_ft,
        out_ft_counts=out_ft_counts,
        combined=True,
    )

    assert 7 in out_force["poly"]
    assert 7 in out_torque["poly"]
    assert 7 in out_ft["poly"]
    assert out_counts["poly"][7] == 1
    assert out_ft_counts["poly"][7] == 1


def test_build_ua_vectors_customised_axes_uses_get_UA_axes():
    node = FrameCovarianceNode()

    bead0 = MagicMock()
    bead1 = MagicMock()
    bead_groups = [bead0, bead1]
    residue_atoms = MagicMock()

    axes_manager = MagicMock()
    axes_manager.get_UA_axes.side_effect = [
        (
            np.eye(3),
            np.eye(3) * 2,
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ),
        (
            np.eye(3) * 3,
            np.eye(3) * 4,
            np.array([7.0, 8.0, 9.0]),
            np.array([10.0, 11.0, 12.0]),
        ),
    ]

    node._ft.get_weighted_forces = MagicMock(
        side_effect=[np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    )
    node._ft.get_weighted_torques = MagicMock(
        side_effect=[np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 0.0])]
    )

    force_vecs, torque_vecs = node._build_ua_vectors(
        bead_groups=bead_groups,
        residue_atoms=residue_atoms,
        axes_manager=axes_manager,
        box=np.array([10.0, 10.0, 10.0]),
        force_partitioning=1.5,
        customised_axes=True,
        is_highest=True,
    )

    assert len(force_vecs) == 2
    assert len(torque_vecs) == 2

    np.testing.assert_allclose(force_vecs[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(force_vecs[1], np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(torque_vecs[0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(torque_vecs[1], np.array([0.0, 2.0, 0.0]))

    assert axes_manager.get_UA_axes.call_count == 2
    axes_manager.get_UA_axes.assert_any_call(residue_atoms, 0)
    axes_manager.get_UA_axes.assert_any_call(residue_atoms, 1)

    assert node._ft.get_weighted_forces.call_count == 2
    assert node._ft.get_weighted_torques.call_count == 2


def test_build_ua_vectors_vanilla_axes_uses_make_whole_and_vanilla_axes():
    node = FrameCovarianceNode()

    bead0 = MagicMock()
    bead1 = MagicMock()
    bead0.center_of_mass.return_value = np.array([1.0, 1.0, 1.0])
    bead1.center_of_mass.return_value = np.array([2.0, 2.0, 2.0])

    bead_groups = [bead0, bead1]
    residue_atoms = MagicMock()
    residue_atoms.principal_axes.return_value = np.eye(3)

    axes_manager = MagicMock()
    axes_manager.get_vanilla_axes.side_effect = [
        (np.eye(3) * 2, np.array([3.0, 4.0, 5.0])),
        (np.eye(3) * 3, np.array([6.0, 7.0, 8.0])),
    ]

    node._ft.get_weighted_forces = MagicMock(
        side_effect=[np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    )
    node._ft.get_weighted_torques = MagicMock(
        side_effect=[np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 0.0])]
    )

    with patch("CodeEntropy.levels.nodes.covariance.make_whole") as make_whole:
        force_vecs, torque_vecs = node._build_ua_vectors(
            bead_groups=bead_groups,
            residue_atoms=residue_atoms,
            axes_manager=axes_manager,
            box=np.array([20.0, 20.0, 20.0]),
            force_partitioning=2.0,
            customised_axes=False,
            is_highest=False,
        )

    assert len(force_vecs) == 2
    assert len(torque_vecs) == 2

    assert make_whole.call_count == 4
    make_whole.assert_any_call(residue_atoms)
    make_whole.assert_any_call(bead0)
    make_whole.assert_any_call(bead1)

    assert residue_atoms.principal_axes.call_count == 2
    assert axes_manager.get_vanilla_axes.call_count == 2
    bead0.center_of_mass.assert_called_once_with(unwrap=True)
    bead1.center_of_mass.assert_called_once_with(unwrap=True)

    np.testing.assert_allclose(force_vecs[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(force_vecs[1], np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(torque_vecs[0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(torque_vecs[1], np.array([0.0, 2.0, 0.0]))
