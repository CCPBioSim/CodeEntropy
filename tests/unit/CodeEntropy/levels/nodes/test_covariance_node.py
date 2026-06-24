from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from CodeEntropy.levels.nodes.covariance import FrameCovarianceNode


class FakeAtomGroup:
    """Small AtomGroup-like object for covariance-node tests."""

    def __init__(self, name="ag", *, length=1):
        self.name = name
        self._length = length
        self.indices = np.arange(length)

    def __len__(self):
        return self._length

    def principal_axes(self):
        return np.eye(3)

    def center_of_mass(self, unwrap=False):
        return np.array([1.0, 2.0, 3.0])


class FakeResidue:
    """Small residue-like object."""

    def __init__(self, atoms=None):
        self.atoms = atoms or FakeAtomGroup("residue-atoms")


class FakeMolecule:
    """Small molecule-like fragment."""

    def __init__(self, n_residues=1):
        self.atoms = FakeAtomGroup("mol-atoms")
        self.residues = [FakeResidue() for _ in range(n_residues)]


class FakeAtoms:
    """Container supporting u.atoms.fragments and u.atoms[index_array]."""

    def __init__(self, fragments, returned_groups=None):
        self.fragments = fragments
        self.returned_groups = list(returned_groups or [])

    def __getitem__(self, index):
        if self.returned_groups:
            return self.returned_groups.pop(0)
        return FakeAtomGroup(f"group-{index}", length=1)


class FakeUniverse:
    """Small universe-like object."""

    def __init__(self, fragments, *, dimensions=None, returned_groups=None):
        self.atoms = FakeAtoms(fragments, returned_groups=returned_groups)
        if dimensions is not None:
            self.dimensions = dimensions


def _args(
    *,
    force_partitioning=0.5,
    combined_forcetorque=False,
    customised_axes=False,
):
    return SimpleNamespace(
        force_partitioning=force_partitioning,
        combined_forcetorque=combined_forcetorque,
        customised_axes=customised_axes,
    )


def test_run_processes_all_levels_and_writes_frame_covariance():
    node = FrameCovarianceNode()
    node._process_united_atom = MagicMock()
    node._process_residue = MagicMock()
    node._process_polymer = MagicMock()

    mol = FakeMolecule()
    universe = FakeUniverse([mol], dimensions=np.array([10.0, 20.0, 30.0, 90.0]))
    axes_manager = object()
    axes_topology = object()

    ctx = {
        "shared": {
            "reduced_universe": universe,
            "groups": {7: [0]},
            "levels": [["united_atom", "residue", "polymer"]],
            "beads": {},
            "args": _args(combined_forcetorque=True, customised_axes=True),
            "axes_manager": axes_manager,
            "axes_topology": axes_topology,
        }
    }

    result = node.run(ctx)

    assert ctx["frame_covariance"] is result
    assert set(result) == {"force", "torque", "forcetorque"}

    node._process_united_atom.assert_called_once()
    node._process_residue.assert_called_once()
    node._process_polymer.assert_called_once()

    ua_kwargs = node._process_united_atom.call_args.kwargs
    assert ua_kwargs["u"] is universe
    assert ua_kwargs["mol"] is mol
    assert ua_kwargs["mol_id"] == 0
    assert ua_kwargs["group_id"] == 7
    assert ua_kwargs["axes_manager"] is axes_manager
    assert ua_kwargs["axes_topology"] is axes_topology
    assert ua_kwargs["force_partitioning"] == 0.5
    assert ua_kwargs["customised_axes"] is True
    assert ua_kwargs["is_highest"] is False

    res_kwargs = node._process_residue.call_args.kwargs
    assert res_kwargs["axes_topology"] is axes_topology


def test_run_omits_forcetorque_when_combined_is_false():
    node = FrameCovarianceNode()
    node._process_united_atom = MagicMock()
    node._process_residue = MagicMock()
    node._process_polymer = MagicMock()

    ctx = {
        "shared": {
            "reduced_universe": FakeUniverse([FakeMolecule()]),
            "groups": {0: [0]},
            "levels": [["residue"]],
            "beads": {},
            "args": _args(combined_forcetorque=False),
        }
    }

    result = node.run(ctx)

    assert set(result) == {"force", "torque"}


def test_process_united_atom_updates_outputs_and_molcount():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    bead_group = FakeAtomGroup("ua", length=1)
    universe = FakeUniverse([mol], returned_groups=[bead_group])

    node._build_ua_vectors = MagicMock(
        return_value=([np.array([1.0, 0.0, 0.0])], [np.array([0.0, 1.0, 0.0])])
    )
    node._ft.compute_frame_covariance = MagicMock(
        return_value=(np.eye(3), 2.0 * np.eye(3))
    )

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    molcount = {}

    node._process_united_atom(
        u=universe,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={(0, "united_atom", 0): [np.array([0])]},
        axes_manager="axes",
        axes_topology=None,
        box=None,
        force_partitioning=0.5,
        customised_axes=False,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        molcount=molcount,
    )

    np.testing.assert_allclose(out_force["ua"][(7, 0)], np.eye(3))
    np.testing.assert_allclose(out_torque["ua"][(7, 0)], 2.0 * np.eye(3))
    assert molcount[(7, 0)] == 1


def test_process_united_atom_returns_when_no_beads_or_empty_atom_groups():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}

    node._process_united_atom(
        u=FakeUniverse([mol]),
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={},
        axes_manager=None,
        axes_topology=None,
        box=None,
        force_partitioning=0.5,
        customised_axes=False,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        molcount={},
    )

    assert out_force["ua"] == {}

    empty_group_universe = FakeUniverse(
        [mol], returned_groups=[FakeAtomGroup(length=0)]
    )
    node._process_united_atom(
        u=empty_group_universe,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={(0, "united_atom", 0): [np.array([0])]},
        axes_manager=None,
        axes_topology=None,
        box=None,
        force_partitioning=0.5,
        customised_axes=False,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        molcount={},
    )

    assert out_force["ua"] == {}


def test_process_residue_updates_outputs_and_combined_ft():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    universe = FakeUniverse([mol], returned_groups=[FakeAtomGroup("residue", length=1)])

    force_vecs = [np.array([1.0, 0.0, 0.0])]
    torque_vecs = [np.array([0.0, 1.0, 0.0])]
    node._build_residue_vectors = MagicMock(return_value=(force_vecs, torque_vecs))
    node._ft.compute_frame_covariance = MagicMock(
        return_value=(np.eye(3), 2.0 * np.eye(3))
    )
    node._build_ft_block = MagicMock(return_value=np.ones((6, 6)))

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_ft = {"ua": {}, "res": {}, "poly": {}}

    node._process_residue(
        u=universe,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={(0, "residue"): [np.array([0])]},
        axes_manager="axes",
        axes_topology=None,
        box=None,
        customised_axes=True,
        force_partitioning=0.5,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_ft=out_ft,
        molcount={},
        combined=True,
    )

    np.testing.assert_allclose(out_force["res"][7], np.eye(3))
    np.testing.assert_allclose(out_torque["res"][7], 2.0 * np.eye(3))
    np.testing.assert_allclose(out_ft["res"][7], np.ones((6, 6)))


def test_process_residue_returns_when_no_beads_or_empty_groups():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}

    node._process_residue(
        u=FakeUniverse([mol]),
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={},
        axes_manager=None,
        axes_topology=None,
        box=None,
        customised_axes=False,
        force_partitioning=0.5,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_ft=None,
        molcount={},
        combined=False,
    )

    assert out_force["res"] == {}

    empty_universe = FakeUniverse([mol], returned_groups=[FakeAtomGroup(length=0)])
    node._process_residue(
        u=empty_universe,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={(0, "residue"): [np.array([0])]},
        axes_manager=None,
        axes_topology=None,
        box=None,
        customised_axes=False,
        force_partitioning=0.5,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_ft=None,
        molcount={},
        combined=False,
    )

    assert out_force["res"] == {}


def test_process_polymer_updates_outputs_and_combined_ft():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    bead = FakeAtomGroup("polymer", length=1)
    universe = FakeUniverse([mol], returned_groups=[bead])

    node._get_polymer_axes = MagicMock(
        return_value=(np.eye(3), np.eye(3), np.zeros(3), np.ones(3))
    )
    node._ft.get_weighted_forces = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
    node._ft.get_weighted_torques = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))
    node._ft.compute_frame_covariance = MagicMock(
        return_value=(np.eye(3), 2.0 * np.eye(3))
    )
    node._build_ft_block = MagicMock(return_value=np.ones((6, 6)))

    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}
    out_ft = {"ua": {}, "res": {}, "poly": {}}

    node._process_polymer(
        u=universe,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={(0, "polymer"): [np.array([0])]},
        axes_manager="axes",
        box=None,
        force_partitioning=0.5,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_ft=out_ft,
        molcount={},
        combined=True,
    )

    np.testing.assert_allclose(out_force["poly"][7], np.eye(3))
    np.testing.assert_allclose(out_torque["poly"][7], 2.0 * np.eye(3))
    np.testing.assert_allclose(out_ft["poly"][7], np.ones((6, 6)))


def test_process_polymer_returns_when_no_beads_or_empty_groups():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    out_force = {"ua": {}, "res": {}, "poly": {}}
    out_torque = {"ua": {}, "res": {}, "poly": {}}

    node._process_polymer(
        u=FakeUniverse([mol]),
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={},
        axes_manager=None,
        box=None,
        force_partitioning=0.5,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_ft=None,
        molcount={},
        combined=False,
    )

    assert out_force["poly"] == {}

    empty_universe = FakeUniverse([mol], returned_groups=[FakeAtomGroup(length=0)])
    node._process_polymer(
        u=empty_universe,
        mol=mol,
        mol_id=0,
        group_id=7,
        beads={(0, "polymer"): [np.array([0])]},
        axes_manager=None,
        box=None,
        force_partitioning=0.5,
        is_highest=True,
        out_force=out_force,
        out_torque=out_torque,
        out_ft=None,
        molcount={},
        combined=False,
    )

    assert out_force["poly"] == {}


def test_build_ua_vectors_uses_customised_axes():
    node = FrameCovarianceNode()
    axes_manager = MagicMock()
    axes_manager.get_UA_axes.return_value = (
        np.eye(3),
        2.0 * np.eye(3),
        np.ones(3),
        np.array([1.0, 2.0, 3.0]),
    )
    node._ft.get_weighted_forces = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
    node._ft.get_weighted_torques = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))

    force_vecs, torque_vecs = node._build_ua_vectors(
        u=FakeUniverse([]),
        mol_id=0,
        local_res_i=0,
        bead_groups=[FakeAtomGroup("ua")],
        residue_atoms=FakeAtomGroup("res"),
        axes_manager=axes_manager,
        axes_topology=None,
        box=None,
        force_partitioning=0.5,
        customised_axes=True,
        is_highest=True,
    )

    assert len(force_vecs) == 1
    assert len(torque_vecs) == 1
    axes_manager.get_UA_axes.assert_called_once()


def test_build_ua_vectors_uses_cached_axes_topology_when_available():
    node = FrameCovarianceNode()
    axes_manager = MagicMock()

    u = FakeUniverse([])
    ua_topology = object()
    axes_topology = SimpleNamespace(ua={(3, 4, 0): ua_topology})

    axes_manager.get_UA_axes_from_topology.return_value = (
        np.eye(3),
        2.0 * np.eye(3),
        np.ones(3),
        np.array([1.0, 2.0, 3.0]),
    )
    node._ft.get_weighted_forces = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
    node._ft.get_weighted_torques = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))

    force_vecs, torque_vecs = node._build_ua_vectors(
        u=u,
        mol_id=3,
        local_res_i=4,
        bead_groups=[FakeAtomGroup("ua")],
        residue_atoms=FakeAtomGroup("res"),
        axes_manager=axes_manager,
        axes_topology=axes_topology,
        box=None,
        force_partitioning=0.5,
        customised_axes=True,
        is_highest=True,
    )

    assert len(force_vecs) == 1
    assert len(torque_vecs) == 1

    called_kwargs = axes_manager.get_UA_axes_from_topology.call_args.kwargs
    assert called_kwargs["u"] is u
    assert called_kwargs["topology"] is ua_topology
    assert called_kwargs["box"] is None
    assert called_kwargs["residue_atoms"].name == "res"

    axes_manager.get_UA_axes.assert_not_called()


def test_build_ua_vectors_uses_vanilla_axes_when_not_customised():
    node = FrameCovarianceNode()
    axes_manager = MagicMock()
    axes_manager.get_vanilla_axes.return_value = (
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),
    )
    node._ft.get_weighted_forces = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
    node._ft.get_weighted_torques = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))

    with patch("CodeEntropy.levels.nodes.covariance.make_whole") as make_whole:
        node._build_ua_vectors(
            u=FakeUniverse([]),
            mol_id=0,
            local_res_i=0,
            bead_groups=[FakeAtomGroup("ua")],
            residue_atoms=FakeAtomGroup("res"),
            axes_manager=axes_manager,
            axes_topology=None,
            box=None,
            force_partitioning=0.5,
            customised_axes=False,
            is_highest=False,
        )

    assert make_whole.call_count == 2
    axes_manager.get_vanilla_axes.assert_called_once()


def test_build_residue_vectors_uses_residue_axes():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    axes_manager = MagicMock()

    node._get_residue_axes = MagicMock(
        return_value=(np.eye(3), np.eye(3), np.zeros(3), np.ones(3))
    )
    node._ft.get_weighted_forces = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
    node._ft.get_weighted_torques = MagicMock(return_value=np.array([0.0, 1.0, 0.0]))

    force_vecs, torque_vecs = node._build_residue_vectors(
        u=FakeUniverse([mol]),
        mol=mol,
        mol_id=0,
        bead_groups=[FakeAtomGroup("res")],
        axes_manager=axes_manager,
        axes_topology=None,
        box=None,
        customised_axes=True,
        force_partitioning=0.5,
        is_highest=True,
    )

    assert len(force_vecs) == 1
    assert len(torque_vecs) == 1
    node._get_residue_axes.assert_called_once()


def test_get_residue_axes_customised_uses_cached_topology_when_available():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    axes_manager = MagicMock()
    expected = (np.eye(3), np.eye(3) * 2.0, np.zeros(3), np.ones(3))
    residue_topology = object()
    axes_topology = SimpleNamespace(residue={(3, 0): residue_topology})
    axes_manager.get_residue_axes_from_topology.return_value = expected

    result = node._get_residue_axes(
        u=FakeUniverse([mol]),
        mol=mol,
        mol_id=3,
        bead=FakeAtomGroup("res"),
        local_res_i=0,
        axes_manager=axes_manager,
        axes_topology=axes_topology,
        box=None,
        customised_axes=True,
    )

    assert result == expected
    called_kwargs = axes_manager.get_residue_axes_from_topology.call_args.kwargs
    assert called_kwargs["topology"] is residue_topology
    assert called_kwargs["residue_atoms"] is mol.residues[0].atoms
    axes_manager.get_residue_axes.assert_not_called()


def test_get_residue_axes_customised_delegates_to_axes_manager():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    axes_manager = MagicMock()
    expected = (np.eye(3), np.eye(3), np.zeros(3), np.ones(3))
    axes_manager.get_residue_axes.return_value = expected

    assert (
        node._get_residue_axes(
            u=FakeUniverse([mol]),
            mol=mol,
            mol_id=0,
            bead=FakeAtomGroup("res"),
            local_res_i=0,
            axes_manager=axes_manager,
            axes_topology=None,
            box=None,
            customised_axes=True,
        )
        == expected
    )

    axes_manager.get_residue_axes.assert_called_once_with(
        mol,
        0,
        residue=mol.residues[0].atoms,
    )


def test_get_residue_axes_vanilla_uses_make_whole_and_vanilla_axes():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    bead = FakeAtomGroup("res")
    axes_manager = MagicMock()
    axes_manager.get_vanilla_axes.return_value = (
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),
    )

    with patch("CodeEntropy.levels.nodes.covariance.make_whole") as make_whole:
        trans_axes, rot_axes, center, moi = node._get_residue_axes(
            u=FakeUniverse([mol]),
            mol=mol,
            mol_id=0,
            bead=bead,
            local_res_i=0,
            axes_manager=axes_manager,
            axes_topology=None,
            box=None,
            customised_axes=False,
        )

    assert make_whole.call_count == 2
    np.testing.assert_allclose(trans_axes, np.eye(3))
    np.testing.assert_allclose(rot_axes, np.eye(3))
    np.testing.assert_allclose(center, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(moi, np.array([1.0, 2.0, 3.0]))


def test_get_polymer_axes_uses_make_whole_and_vanilla_axes():
    node = FrameCovarianceNode()
    mol = FakeMolecule(n_residues=1)
    bead = FakeAtomGroup("poly")
    axes_manager = MagicMock()
    axes_manager.get_vanilla_axes.return_value = (
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),
    )

    with patch("CodeEntropy.levels.nodes.covariance.make_whole") as make_whole:
        trans_axes, rot_axes, center, moi = node._get_polymer_axes(
            mol=mol,
            bead=bead,
            axes_manager=axes_manager,
        )

    assert make_whole.call_count == 2
    np.testing.assert_allclose(trans_axes, np.eye(3))
    np.testing.assert_allclose(rot_axes, np.eye(3))
    np.testing.assert_allclose(center, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(moi, np.array([1.0, 2.0, 3.0]))


def test_get_shared_requires_shared_key():
    with pytest.raises(KeyError, match="ctx\\['shared'\\]"):
        FrameCovarianceNode._get_shared({})


def test_try_get_box_returns_dimensions_or_none():
    assert np.allclose(
        FrameCovarianceNode._try_get_box(SimpleNamespace(dimensions=[1, 2, 3, 90])),
        np.array([1.0, 2.0, 3.0]),
    )
    assert FrameCovarianceNode._try_get_box(object()) is None


def test_inc_mean_copies_first_value_and_updates_existing_mean():
    new = np.array([1.0, 2.0])
    out = FrameCovarianceNode._inc_mean(None, new, n=1)
    new[0] = 99.0
    assert out[0] == 1.0

    np.testing.assert_allclose(
        FrameCovarianceNode._inc_mean(np.array([2.0, 2.0]), np.array([4.0, 0.0]), n=2),
        np.array([3.0, 1.0]),
    )


def test_build_ft_block_builds_symmetric_block_matrix():
    force_vecs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    torque_vecs = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0])]

    out = FrameCovarianceNode._build_ft_block(force_vecs, torque_vecs)

    assert out.shape == (12, 12)
    np.testing.assert_allclose(out[:6, 6:], out[6:, :6].T)


def test_build_ft_block_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same length"):
        FrameCovarianceNode._build_ft_block([np.ones(3)], [])

    with pytest.raises(ValueError, match="No bead vectors"):
        FrameCovarianceNode._build_ft_block([], [])

    with pytest.raises(ValueError, match="length 3"):
        FrameCovarianceNode._build_ft_block([np.ones(2)], [np.ones(3)])
