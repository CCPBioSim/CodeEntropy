from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pytest

from CodeEntropy.levels.dihedrals.topology import (
    DihedralTopologyDiscovery,
    MoleculeDihedralTopology,
)


@dataclass
class FakeAtom:
    """Small test double for an MDAnalysis Atom."""

    index: int
    bonded_atoms: Any = None


class FakeUniverseAtoms:
    """Indexable fake universe atom container."""

    def __init__(self, universe: FakeUniverse) -> None:
        self._universe = universe

    def __getitem__(self, indices: int | list[int] | tuple[int, ...]) -> Any:
        if isinstance(indices, int):
            return self._universe.atom_by_index[int(indices)]

        return FakeAtomGroup(
            [self._universe.atom_by_index[int(index)] for index in indices],
            universe=self._universe,
        )


class FakeUniverse:
    """Small fake universe supporting ``universe.atoms[indices]``."""

    def __init__(self, atoms: list[FakeAtom]) -> None:
        self.atom_by_index = {int(atom.index): atom for atom in atoms}
        self.atoms = FakeUniverseAtoms(self)


class FakeAtomGroup:
    """Small AtomGroup-like test double for topology discovery tests."""

    def __init__(
        self,
        atoms: list[FakeAtom],
        *,
        residues: list[Any] | None = None,
        dihedrals: list[Any] | None = None,
        select_map: dict[str, Any] | None = None,
        universe: FakeUniverse | None = None,
    ) -> None:
        self._atoms = list(atoms)
        self.residues = list(residues or [])
        self.dihedrals = list(dihedrals or [])
        self._select_map = dict(select_map or {})
        self.universe = universe

    @property
    def atoms(self) -> FakeAtomGroup:
        return self

    @property
    def indices(self) -> list[int]:
        return [int(atom.index) for atom in self._atoms]

    def __iter__(self):
        return iter(self._atoms)

    def __len__(self) -> int:
        return len(self._atoms)

    def __add__(self, other: FakeAtomGroup) -> FakeAtomGroup:
        return FakeAtomGroup(
            self._atoms + other._atoms,
            universe=self.universe or other.universe,
        )

    def select_atoms(self, select_string: str, updating: bool = False) -> Any:
        if select_string not in self._select_map:
            raise AssertionError(f"Unexpected selection: {select_string!r}")
        return self._select_map[select_string]


@dataclass
class FakeResidue:
    """Small residue test double."""

    atoms: FakeAtomGroup


@dataclass
class FakeDihedral:
    """Small dihedral topology object."""

    atoms: FakeAtomGroup


def _make_discovery(
    universe_operations: Any | None = None,
) -> DihedralTopologyDiscovery:
    discovery = DihedralTopologyDiscovery()
    discovery._universe_operations = universe_operations or Mock()
    return discovery


def test_molecule_dihedral_topology_stores_expected_fields() -> None:
    topology = MoleculeDihedralTopology(
        group_id=1,
        molecule_id=2,
        molecule_order=3,
        num_residues=4,
        ua_dihedrals_by_residue={0: ["ua"]},
        residue_dihedrals=["res"],
    )

    assert topology.group_id == 1
    assert topology.molecule_id == 2
    assert topology.molecule_order == 3
    assert topology.num_residues == 4
    assert topology.ua_dihedrals_by_residue == {0: ["ua"]}
    assert topology.residue_dihedrals == ["res"]


def test_extract_topology_fragment_uses_lightweight_atomgroup_helper() -> None:
    universe_operations = Mock()
    universe_operations.extract_fragment_atomgroup.return_value = "fragment_atomgroup"
    universe_operations.extract_fragment.return_value = "heavy_fragment"

    discovery = _make_discovery(universe_operations)

    result = discovery._extract_topology_fragment("universe", 5)

    assert result == "fragment_atomgroup"
    universe_operations.extract_fragment_atomgroup.assert_called_once_with(
        "universe",
        5,
    )
    universe_operations.extract_fragment.assert_not_called()


def test_select_heavy_residue_builds_expected_selections() -> None:
    discovery = _make_discovery()

    residue_atoms = Mock()
    residue_atoms.indices = [10, 11, 12, 13]
    residue = SimpleNamespace(atoms=residue_atoms)

    heavy_atoms = object()
    residue_container = Mock()
    residue_container.select_atoms.return_value = heavy_atoms

    mol = Mock()
    mol.residues = [residue]
    mol.select_atoms.return_value = residue_container

    result = discovery._select_heavy_residue(mol, 0)

    assert result is heavy_atoms
    mol.select_atoms.assert_called_once_with("index 10:13", updating=False)
    residue_container.select_atoms.assert_called_once_with(
        "prop mass > 1.1",
        updating=False,
    )
    discovery._universe_operations.select_atoms.assert_not_called()


def test_get_dihedrals_united_atom_collects_atoms_from_dihedral_objects() -> None:
    discovery = _make_discovery()

    atoms = [FakeAtom(index) for index in range(1, 8)]
    universe = FakeUniverse(atoms)

    valid_dihedral_atoms = FakeAtomGroup(
        [atoms[0], atoms[1], atoms[2], atoms[3]],
        universe=universe,
    )
    outside_selection_atoms = FakeAtomGroup(
        [atoms[0], atoms[1], atoms[2], atoms[6]],
        universe=universe,
    )
    wrong_size_atoms = FakeAtomGroup(
        [atoms[0], atoms[1], atoms[2]],
        universe=universe,
    )

    selected_heavy_atoms = FakeAtomGroup(
        [atoms[0], atoms[1], atoms[2], atoms[3], atoms[4]],
        dihedrals=[
            FakeDihedral(valid_dihedral_atoms),
            FakeDihedral(outside_selection_atoms),
            FakeDihedral(wrong_size_atoms),
        ],
        universe=universe,
    )

    result = discovery._get_dihedrals(selected_heavy_atoms, "united_atom")

    assert result == [valid_dihedral_atoms]


def test_get_dihedrals_united_atom_returns_empty_when_no_valid_dihedrals() -> None:
    discovery = _make_discovery()

    atoms = [FakeAtom(index) for index in range(1, 5)]
    universe = FakeUniverse(atoms)

    invalid_dihedral_atoms = FakeAtomGroup(
        [atoms[0], atoms[1], atoms[2]],
        universe=universe,
    )
    selected_heavy_atoms = FakeAtomGroup(
        atoms,
        dihedrals=[FakeDihedral(invalid_dihedral_atoms)],
        universe=universe,
    )

    assert discovery._get_dihedrals(selected_heavy_atoms, "united_atom") == []


def test_get_dihedrals_residue_builds_one_dihedral_when_four_residues() -> None:
    discovery = _make_discovery()

    atom1 = FakeAtom(10)
    atom2 = FakeAtom(20)
    atom3 = FakeAtom(30)
    atom4 = FakeAtom(40)

    universe = FakeUniverse([atom1, atom2, atom3, atom4])

    group1 = FakeAtomGroup([atom1], universe=universe)
    group2 = FakeAtomGroup([atom2], universe=universe)
    group3 = FakeAtomGroup([atom3], universe=universe)
    group4 = FakeAtomGroup([atom4], universe=universe)

    atom1.bonded_atoms = group2
    atom2.bonded_atoms = group1
    atom3.bonded_atoms = group4
    atom4.bonded_atoms = group3

    mol = FakeAtomGroup(
        [atom1, atom2, atom3, atom4],
        residues=[
            FakeResidue(group1),
            FakeResidue(group2),
            FakeResidue(group3),
            FakeResidue(group4),
        ],
        universe=universe,
    )

    result = discovery._get_dihedrals(mol, "residue")

    assert len(result) == 1
    assert result[0].indices == [10, 20, 30, 40]


def test_get_dihedrals_residue_skips_invalid_four_residue_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    discovery = _make_discovery()

    atom1 = FakeAtom(10)
    atom2 = FakeAtom(20)
    atom3 = FakeAtom(30)
    atom4 = FakeAtom(40)

    universe = FakeUniverse([atom1, atom2, atom3, atom4])

    group1 = FakeAtomGroup([atom1], universe=universe)
    group2 = FakeAtomGroup([atom2], universe=universe)
    group3 = FakeAtomGroup([atom3], universe=universe)
    group4 = FakeAtomGroup([atom4], universe=universe)

    atom1.bonded_atoms = group2
    atom2.bonded_atoms = group1
    atom3.bonded_atoms = FakeAtomGroup([], universe=universe)
    atom4.bonded_atoms = FakeAtomGroup([], universe=universe)

    mol = FakeAtomGroup(
        [atom1, atom2, atom3, atom4],
        residues=[
            FakeResidue(group1),
            FakeResidue(group2),
            FakeResidue(group3),
            FakeResidue(group4),
        ],
        universe=universe,
    )

    result = discovery._get_dihedrals(mol, "residue")

    assert result == []
    assert "Skipping residue-level dihedral" in caplog.text


def test_get_dihedrals_residue_returns_empty_when_fewer_than_four_residues() -> None:
    discovery = _make_discovery()

    atoms = [FakeAtom(1), FakeAtom(2), FakeAtom(3)]
    universe = FakeUniverse(atoms)
    groups = [FakeAtomGroup([atom], universe=universe) for atom in atoms]

    mol = FakeAtomGroup(
        atoms,
        residues=[FakeResidue(group) for group in groups],
        universe=universe,
    )

    assert discovery._get_dihedrals(mol, "residue") == []


def test_atoms_in_source_bonded_to_target_returns_matching_source_atoms() -> None:
    atom1 = FakeAtom(1)
    atom2 = FakeAtom(2)
    atom3 = FakeAtom(3)

    universe = FakeUniverse([atom1, atom2, atom3])

    source_group = FakeAtomGroup([atom1, atom3], universe=universe)
    target_group = FakeAtomGroup([atom2], universe=universe)

    atom1.bonded_atoms = target_group
    atom3.bonded_atoms = FakeAtomGroup([], universe=universe)

    source_residue = FakeResidue(source_group)
    target_residue = FakeResidue(target_group)

    result = DihedralTopologyDiscovery._atoms_in_source_bonded_to_target(
        source_residue,
        target_residue,
    )

    assert result.indices == [1]


def test_atoms_in_source_bonded_to_target_returns_empty_when_atom_has_no_bonds() -> (
    None
):
    atom1 = FakeAtom(1)
    atom2 = FakeAtom(2)

    universe = FakeUniverse([atom1, atom2])

    source_residue = FakeResidue(FakeAtomGroup([atom1], universe=universe))
    target_residue = FakeResidue(FakeAtomGroup([atom2], universe=universe))

    result = DihedralTopologyDiscovery._atoms_in_source_bonded_to_target(
        source_residue,
        target_residue,
    )

    assert result.indices == []


def test_discover_group_dihedral_topology_builds_one_entry_per_molecule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe_operations = Mock()

    molecule0 = SimpleNamespace(label="molecule0", residues=[object(), object()])
    molecule1 = SimpleNamespace(label="molecule1", residues=[object()])

    universe_operations.extract_fragment_atomgroup.side_effect = [
        molecule0,
        molecule1,
    ]

    discovery = _make_discovery(universe_operations)

    def fake_select_heavy_residue(mol: Any, res_id: int) -> str:
        return f"{mol.label}-heavy-{res_id}"

    def fake_get_dihedrals(data_container: Any, level: str) -> list[str]:
        if level == "united_atom":
            return [f"ua-{data_container}"]
        return [f"res-{data_container.label}"]

    monkeypatch.setattr(
        discovery,
        "_select_heavy_residue",
        fake_select_heavy_residue,
    )
    monkeypatch.setattr(
        discovery,
        "_get_dihedrals",
        fake_get_dihedrals,
    )

    topologies = discovery._discover_group_dihedral_topology(
        data_container="universe",
        group_id=9,
        molecules=[100, 200],
        level_list=["united_atom", "residue"],
    )

    assert topologies == [
        MoleculeDihedralTopology(
            group_id=9,
            molecule_id=100,
            molecule_order=0,
            num_residues=2,
            ua_dihedrals_by_residue={
                0: ["ua-molecule0-heavy-0"],
                1: ["ua-molecule0-heavy-1"],
            },
            residue_dihedrals=["res-molecule0"],
        ),
        MoleculeDihedralTopology(
            group_id=9,
            molecule_id=200,
            molecule_order=1,
            num_residues=1,
            ua_dihedrals_by_residue={
                0: ["ua-molecule1-heavy-0"],
            },
            residue_dihedrals=["res-molecule1"],
        ),
    ]

    assert universe_operations.extract_fragment_atomgroup.call_args_list == [
        call("universe", 100),
        call("universe", 200),
    ]


def test_discover_group_dihedral_topology_respects_enabled_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe_operations = Mock()

    molecule = SimpleNamespace(label="molecule", residues=[object(), object()])
    universe_operations.extract_fragment_atomgroup.return_value = molecule

    discovery = _make_discovery(universe_operations)

    select_heavy_residue = Mock(return_value="heavy-residue")
    get_dihedrals = Mock(return_value=["residue-dihedral"])

    monkeypatch.setattr(discovery, "_select_heavy_residue", select_heavy_residue)
    monkeypatch.setattr(discovery, "_get_dihedrals", get_dihedrals)

    topologies = discovery._discover_group_dihedral_topology(
        data_container="universe",
        group_id=1,
        molecules=[0],
        level_list=["residue"],
    )

    assert topologies == [
        MoleculeDihedralTopology(
            group_id=1,
            molecule_id=0,
            molecule_order=0,
            num_residues=2,
            ua_dihedrals_by_residue={},
            residue_dihedrals=["residue-dihedral"],
        )
    ]

    select_heavy_residue.assert_not_called()
    get_dihedrals.assert_called_once_with(molecule, "residue")
