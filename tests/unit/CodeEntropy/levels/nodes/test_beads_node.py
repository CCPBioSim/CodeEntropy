"""Atomic unit tests for bead-definition construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from CodeEntropy.levels.nodes.beads import BuildBeadsNode


class FakeHeavyAtoms:
    """Minimal heavy-atom selection result."""

    def __init__(self, resindices):
        self._atoms = [SimpleNamespace(resindex=resindex) for resindex in resindices]

    def __len__(self):
        return len(self._atoms)

    def __getitem__(self, index):
        return self._atoms[index]


class FakeBead:
    """Minimal AtomGroup-like bead."""

    def __init__(self, indices, *, heavy_resindices=None):
        self.indices = np.asarray(indices, dtype=int)
        self._heavy_resindices = [] if heavy_resindices is None else heavy_resindices

    def __len__(self):
        return int(self.indices.size)

    def select_atoms(self, selection):
        assert selection == "prop mass > 1.1"
        return FakeHeavyAtoms(self._heavy_resindices)


class FakeMolecule:
    """Minimal molecule fragment with residues."""

    def __init__(self, name, residue_indices):
        self.name = name
        self.residues = [
            SimpleNamespace(resindex=resindex) for resindex in residue_indices
        ]


class FakeUniverse:
    """Minimal universe with atom fragments."""

    def __init__(self, fragments):
        self.atoms = SimpleNamespace(fragments=fragments)


class FakeHierarchy:
    """Controlled HierarchyBuilder test double."""

    def __init__(self, bead_map):
        self.bead_map = bead_map
        self.calls = []

    def get_beads(self, mol, level):
        self.calls.append((mol.name, level))
        return self.bead_map.get((mol.name, level), [])


def test_run_builds_all_requested_bead_levels_and_writes_shared_data():
    mol0 = FakeMolecule("mol0", residue_indices=[10, 20])
    mol1 = FakeMolecule("mol1", residue_indices=[30])
    universe = FakeUniverse([mol0, mol1])

    hierarchy = FakeHierarchy(
        {
            ("mol0", "united_atom"): [
                FakeBead([0, 1], heavy_resindices=[10]),
                FakeBead([2], heavy_resindices=[20]),
                FakeBead([], heavy_resindices=[]),
            ],
            ("mol0", "residue"): [FakeBead([0, 1]), FakeBead([2])],
            ("mol0", "polymer"): [FakeBead([0, 1, 2])],
            ("mol1", "residue"): [FakeBead([3, 4])],
        }
    )

    shared_data = {
        "reduced_universe": universe,
        "levels": [
            ["united_atom", "residue", "polymer"],
            ["residue"],
        ],
    }

    result = BuildBeadsNode(hierarchy=hierarchy).run(shared_data)

    beads = shared_data["beads"]
    assert result == {"beads": beads}

    assert hierarchy.calls == [
        ("mol0", "united_atom"),
        ("mol0", "residue"),
        ("mol0", "polymer"),
        ("mol1", "residue"),
    ]

    np.testing.assert_array_equal(beads[(0, "united_atom", 0)][0], np.array([0, 1]))
    np.testing.assert_array_equal(beads[(0, "united_atom", 1)][0], np.array([2]))

    np.testing.assert_array_equal(beads[(0, "residue")][0], np.array([0, 1]))
    np.testing.assert_array_equal(beads[(0, "residue")][1], np.array([2]))
    np.testing.assert_array_equal(beads[(0, "polymer")][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(beads[(1, "residue")][0], np.array([3, 4]))


def test_run_requires_reduced_universe():
    with pytest.raises(KeyError):
        BuildBeadsNode(hierarchy=FakeHierarchy({})).run({"levels": []})


def test_run_requires_levels():
    with pytest.raises(KeyError):
        BuildBeadsNode(hierarchy=FakeHierarchy({})).run(
            {"reduced_universe": FakeUniverse([])}
        )


def test_add_united_atom_beads_creates_empty_bucket_for_missing_residue():
    mol = FakeMolecule("mol", residue_indices=[10, 20, 30])
    hierarchy = FakeHierarchy(
        {
            ("mol", "united_atom"): [
                FakeBead([5], heavy_resindices=[20]),
            ]
        }
    )
    beads = {}

    BuildBeadsNode(hierarchy=hierarchy)._add_united_atom_beads(
        beads=beads,
        mol_id=4,
        mol=mol,
    )

    assert beads[(4, "united_atom", 0)] == []
    np.testing.assert_array_equal(beads[(4, "united_atom", 1)][0], np.array([5]))
    assert beads[(4, "united_atom", 2)] == []


def test_add_residue_beads_logs_error_when_all_beads_are_empty(caplog):
    mol = FakeMolecule("mol", residue_indices=[10])
    hierarchy = FakeHierarchy({("mol", "residue"): [FakeBead([])]})
    beads = {}

    BuildBeadsNode(hierarchy=hierarchy)._add_residue_beads(
        beads=beads,
        mol_id=1,
        mol=mol,
    )

    assert beads[(1, "residue")] == []
    assert "No residue beads kept" in caplog.text


def test_add_polymer_beads_skips_empty_beads():
    mol = FakeMolecule("mol", residue_indices=[10])
    hierarchy = FakeHierarchy({("mol", "polymer"): [FakeBead([]), FakeBead([1, 2])]})
    beads = {}

    BuildBeadsNode(hierarchy=hierarchy)._add_polymer_beads(
        beads=beads,
        mol_id=1,
        mol=mol,
    )

    assert len(beads[(1, "polymer")]) == 1
    np.testing.assert_array_equal(beads[(1, "polymer")][0], np.array([1, 2]))


def test_validate_bead_indices_returns_copy_and_skips_empty_beads(caplog):
    bead = FakeBead([1, 2, 3])
    out = BuildBeadsNode._validate_bead_indices(
        bead,
        mol_id=1,
        level="residue",
        bead_i=0,
    )

    np.testing.assert_array_equal(out, np.array([1, 2, 3]))
    bead.indices[0] = 99
    assert out[0] == 1

    empty = BuildBeadsNode._validate_bead_indices(
        FakeBead([]),
        mol_id=1,
        level="residue",
        bead_i=1,
    )

    assert empty is None
    assert "Empty bead skipped" in caplog.text


def test_infer_local_residue_id_uses_first_heavy_atom_resindex():
    mol = FakeMolecule("mol", residue_indices=[10, 20])

    assert (
        BuildBeadsNode._infer_local_residue_id(
            mol=mol,
            bead=FakeBead([1], heavy_resindices=[20]),
        )
        == 1
    )


def test_infer_local_residue_id_falls_back_to_zero_without_heavy_atoms():
    mol = FakeMolecule("mol", residue_indices=[10, 20])

    assert (
        BuildBeadsNode._infer_local_residue_id(
            mol=mol,
            bead=FakeBead([1], heavy_resindices=[]),
        )
        == 0
    )


def test_infer_local_residue_id_falls_back_to_zero_when_resindex_not_in_molecule():
    mol = FakeMolecule("mol", residue_indices=[10, 20])

    assert (
        BuildBeadsNode._infer_local_residue_id(
            mol=mol,
            bead=FakeBead([1], heavy_resindices=[99]),
        )
        == 0
    )
