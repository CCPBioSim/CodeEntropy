from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from CodeEntropy.levels.nodes.axes_topology import (
    AxesTopology,
    BuildAxesTopologyNode,
    UAAxesTopology,
)


class FakeAtomGroup:
    """Small AtomGroup-like object for axes-topology tests."""

    def __init__(self, atoms=None, *, name="ag"):
        self._atoms = list(atoms or [])
        self.name = name
        self.indices = np.asarray([atom.index for atom in self._atoms], dtype=int)

    def __iter__(self):
        return iter(self._atoms)

    def __len__(self):
        return len(self._atoms)

    def __getitem__(self, index):
        if isinstance(index, (list, tuple, np.ndarray)):
            return FakeAtomGroup([self._atoms[int(i)] for i in index])
        return self._atoms[int(index)]

    def select_atoms(self, selection):
        """Return atoms matching the small mass selections used by the node."""
        if selection == "prop mass > 1.1":
            return FakeAtomGroup([atom for atom in self._atoms if atom.mass > 1.1])
        if selection == "mass 2 to 999":
            return FakeAtomGroup(
                [atom for atom in self._atoms if 2.0 <= atom.mass <= 999.0]
            )
        if selection == "mass 1 to 1.1":
            return FakeAtomGroup(
                [atom for atom in self._atoms if 1.0 <= atom.mass <= 1.1]
            )
        raise AssertionError(f"Unexpected selection: {selection}")


class FakeAtom:
    """Small Atom-like object with mass, index, and bonded atoms."""

    def __init__(self, index, mass, bonded_atoms=None):
        self.index = index
        self.mass = mass
        self.bonded_atoms = bonded_atoms or FakeAtomGroup([])


class FakeResidue:
    """Small residue-like object."""

    def __init__(self, atoms):
        self.atoms = atoms


class FakeMolecule:
    """Small molecule-like object with residues."""

    def __init__(self, residues):
        self.residues = residues


class FakeAtoms:
    """Container supporting fragments and u.atoms[index_array]."""

    def __init__(self, fragments, atom_map):
        self.fragments = fragments
        self._atom_map = dict(atom_map)

    def __getitem__(self, indices):
        if isinstance(indices, np.ndarray):
            return FakeAtomGroup([self._atom_map[int(index)] for index in indices])
        if isinstance(indices, (list, tuple)):
            return FakeAtomGroup([self._atom_map[int(index)] for index in indices])
        return self._atom_map[int(indices)]


class FakeUniverse:
    """Small universe-like object."""

    def __init__(self, fragments, atom_map):
        self.atoms = FakeAtoms(fragments=fragments, atom_map=atom_map)


def _args(*, customised_axes):
    return SimpleNamespace(customised_axes=customised_axes)


def _single_molecule_universe():
    """Build a small molecule containing one residue and one UA bead."""
    hydrogen = FakeAtom(index=2, mass=1.0)
    bonded_heavy = FakeAtom(index=3, mass=12.0)
    heavy = FakeAtom(
        index=1,
        mass=12.0,
        bonded_atoms=FakeAtomGroup([hydrogen, bonded_heavy]),
    )
    other_residue_heavy = FakeAtom(index=4, mass=14.0)

    residue_atoms = FakeAtomGroup([heavy, hydrogen, bonded_heavy, other_residue_heavy])
    residue = FakeResidue(residue_atoms)
    molecule = FakeMolecule([residue])

    atom_map = {
        heavy.index: heavy,
        hydrogen.index: hydrogen,
        bonded_heavy.index: bonded_heavy,
        other_residue_heavy.index: other_residue_heavy,
    }
    universe = FakeUniverse([molecule], atom_map)

    return universe, molecule, heavy, hydrogen, bonded_heavy, other_residue_heavy


def test_ua_axes_topology_dataclass_preserves_arrays():
    topology = UAAxesTopology(
        heavy_atom_index=1,
        ua_atom_indices=np.array([1, 2]),
        ua_all_atom_indices=np.array([1, 3, 2]),
        bonded_heavy_indices=np.array([3]),
        bonded_light_indices=np.array([2]),
        residue_heavy_indices=np.array([1, 3, 4]),
        residue_ua_masses=np.array([13.0, 12.0, 14.0]),
    )

    assert topology.heavy_atom_index == 1
    np.testing.assert_array_equal(topology.ua_atom_indices, np.array([1, 2]))
    np.testing.assert_array_equal(topology.ua_all_atom_indices, np.array([1, 3, 2]))
    np.testing.assert_array_equal(topology.bonded_heavy_indices, np.array([3]))
    np.testing.assert_array_equal(topology.bonded_light_indices, np.array([2]))
    np.testing.assert_array_equal(topology.residue_heavy_indices, np.array([1, 3, 4]))
    np.testing.assert_allclose(topology.residue_ua_masses, np.array([13.0, 12.0, 14.0]))


def test_axes_topology_defaults_to_empty_ua_mapping():
    topology = AxesTopology()

    assert topology.ua == {}


def test_run_writes_empty_topology_when_customised_axes_disabled():
    node = BuildAxesTopologyNode()
    shared_data = {"args": _args(customised_axes=False)}

    result = node.run(shared_data)

    assert isinstance(result["axes_topology"], AxesTopology)
    assert result["axes_topology"].ua == {}
    assert shared_data["axes_topology"] is result["axes_topology"]


def test_run_builds_ua_topology_for_united_atom_levels():
    node = BuildAxesTopologyNode()
    universe, _, heavy, hydrogen, bonded_heavy, other_residue_heavy = (
        _single_molecule_universe()
    )
    shared_data = {
        "args": _args(customised_axes=True),
        "reduced_universe": universe,
        "levels": [["united_atom"]],
        "beads": {(0, "united_atom", 0): [np.array([1, 2])]},
    }

    result = node.run(shared_data)

    axes_topology = result["axes_topology"]
    assert shared_data["axes_topology"] is axes_topology
    assert set(axes_topology.ua) == {(0, 0, 0)}

    ua_topology = axes_topology.ua[(0, 0, 0)]
    assert ua_topology.heavy_atom_index == heavy.index
    np.testing.assert_array_equal(ua_topology.ua_atom_indices, np.array([1, 2]))
    np.testing.assert_array_equal(ua_topology.ua_all_atom_indices, np.array([1, 3, 2]))
    np.testing.assert_array_equal(ua_topology.bonded_heavy_indices, np.array([3]))
    np.testing.assert_array_equal(ua_topology.bonded_light_indices, np.array([2]))
    np.testing.assert_array_equal(
        ua_topology.residue_heavy_indices,
        np.array([heavy.index, bonded_heavy.index, other_residue_heavy.index]),
    )
    np.testing.assert_allclose(
        ua_topology.residue_ua_masses,
        np.array([13.0, 12.0, 14.0]),
    )


def test_run_ignores_molecules_without_united_atom_level():
    node = BuildAxesTopologyNode()
    universe, _, _, _, _, _ = _single_molecule_universe()
    shared_data = {
        "args": _args(customised_axes=True),
        "reduced_universe": universe,
        "levels": [["residue"]],
        "beads": {(0, "united_atom", 0): [np.array([1, 2])]},
    }

    result = node.run(shared_data)

    assert result["axes_topology"].ua == {}
    assert shared_data["axes_topology"].ua == {}


def test_add_ua_topology_skips_residues_without_beads():
    node = BuildAxesTopologyNode()
    universe, molecule, _, _, _, _ = _single_molecule_universe()
    out = {}

    node._add_ua_topology(
        u=universe,
        mol=molecule,
        mol_id=0,
        beads={},
        out=out,
    )

    assert out == {}


def test_add_ua_topology_skips_ua_beads_without_heavy_atoms(caplog):
    node = BuildAxesTopologyNode()
    hydrogen = FakeAtom(index=2, mass=1.0)
    residue = FakeResidue(FakeAtomGroup([hydrogen]))
    molecule = FakeMolecule([residue])
    universe = FakeUniverse([molecule], {2: hydrogen})
    out = {}

    node._add_ua_topology(
        u=universe,
        mol=molecule,
        mol_id=5,
        beads={(5, "united_atom", 0): [np.array([2])]},
        out=out,
    )

    assert out == {}
    assert "Skipping UA axes topology with no heavy atom" in caplog.text


def test_add_ua_topology_handles_multiple_residues_and_ua_beads():
    node = BuildAxesTopologyNode()

    h0 = FakeAtom(index=10, mass=1.0)
    c0 = FakeAtom(index=11, mass=12.0, bonded_atoms=FakeAtomGroup([h0]))
    residue0 = FakeResidue(FakeAtomGroup([c0, h0]))

    h1 = FakeAtom(index=20, mass=1.0)
    c1 = FakeAtom(index=21, mass=12.0, bonded_atoms=FakeAtomGroup([h1]))
    residue1 = FakeResidue(FakeAtomGroup([c1, h1]))

    molecule = FakeMolecule([residue0, residue1])
    universe = FakeUniverse(
        [molecule],
        {
            h0.index: h0,
            c0.index: c0,
            h1.index: h1,
            c1.index: c1,
        },
    )
    out = {}

    node._add_ua_topology(
        u=universe,
        mol=molecule,
        mol_id=3,
        beads={
            (3, "united_atom", 0): [np.array([11, 10])],
            (3, "united_atom", 1): [np.array([21, 20])],
        },
        out=out,
    )

    assert set(out) == {(3, 0, 0), (3, 1, 0)}
    assert out[(3, 0, 0)].heavy_atom_index == 11
    assert out[(3, 1, 0)].heavy_atom_index == 21


def test_split_bonded_atoms_returns_heavy_and_light_atom_groups():
    hydrogen = FakeAtom(index=2, mass=1.0)
    heavy_bonded = FakeAtom(index=3, mass=12.0)
    atom = FakeAtom(
        index=1,
        mass=12.0,
        bonded_atoms=FakeAtomGroup([hydrogen, heavy_bonded]),
    )

    bonded_heavy, bonded_light = BuildAxesTopologyNode._split_bonded_atoms(atom)

    np.testing.assert_array_equal(bonded_heavy.indices, np.array([3]))
    np.testing.assert_array_equal(bonded_light.indices, np.array([2]))


def test_get_ua_masses_from_topology_adds_bonded_hydrogen_masses():
    hydrogen = FakeAtom(index=2, mass=1.0)
    heavy = FakeAtom(index=1, mass=12.0, bonded_atoms=FakeAtomGroup([hydrogen]))
    other_heavy = FakeAtom(index=3, mass=14.0)
    atom_group = FakeAtomGroup([heavy, hydrogen, other_heavy])

    masses = BuildAxesTopologyNode._get_ua_masses_from_topology(atom_group)

    assert masses == [13.0, 14.0]


def test_get_ua_masses_from_topology_handles_atoms_without_bonded_atoms_attribute():
    class AtomWithoutBonds:
        """Small atom-like object without bonded_atoms."""

        def __init__(self, index, mass):
            self.index = index
            self.mass = mass

    heavy = AtomWithoutBonds(index=1, mass=12.0)
    hydrogen = AtomWithoutBonds(index=2, mass=1.0)
    atom_group = FakeAtomGroup([heavy, hydrogen])

    masses = BuildAxesTopologyNode._get_ua_masses_from_topology(atom_group)

    assert masses == [12.0]


def test_get_ua_masses_from_topology_returns_empty_list_for_light_atoms_only():
    hydrogen = FakeAtom(index=2, mass=1.0)
    atom_group = FakeAtomGroup([hydrogen])

    masses = BuildAxesTopologyNode._get_ua_masses_from_topology(atom_group)

    assert masses == []
