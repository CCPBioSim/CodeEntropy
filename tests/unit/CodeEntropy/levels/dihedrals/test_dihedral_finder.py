from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from CodeEntropy.levels.dihedrals.dihedral_finder import DihedralDefinitions


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


class FakeUniverseDihedrals:
    """Indexable fake dihedral list."""

    def __init__(self, universe: FakeUniverse, atoms) -> None:
        self._universe = universe

    def __getitem__(self, atoms) -> Any:
        length = len(atoms)
        if length < 4:
            return []
        else:
            dihedrals = []
            for group in range(4, length, 1):
                dihedral = atoms[group - 4 : group]
                dihedrals.append(dihedral)
            return dihedrals


class FakeUniverse:
    """Small fake universe supporting ``universe.atoms[indices]``."""

    def __init__(self, atoms: list[FakeAtom]) -> None:
        self.indices = {int(atom.index) for atom in atoms}
        self.dihedrals = FakeUniverseDihedrals(self, atoms)
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


def test_method_res_bonds_ua():
    dd = DihedralDefinitions()
    atoms = [FakeAtom(index) for index in range(4)]
    universe = FakeUniverse(atoms)
    atom_group = FakeAtomGroup(atoms, universe=universe)
    dihedral = FakeDihedral(atom_group)

    data_container = FakeAtomGroup(
        atoms,
        dihedrals=[dihedral],
        universe=universe,
    )

    result = dd.method_res_bonds(data_container, "united_atom")

    assert len(result) == 1
    assert result[0] is atom_group
    assert result[0].indices == [0, 1, 2, 3]


def test_method_res_bonds_res():
    dd = DihedralDefinitions()
    atoms = [FakeAtom(index) for index in range(4)]
    universe = FakeUniverse(atoms)

    # Create bonds between consecutive residues.
    atoms[0].bonded_atoms = FakeAtomGroup([atoms[1]], universe=universe)
    atoms[1].bonded_atoms = FakeAtomGroup([atoms[0]], universe=universe)
    atoms[2].bonded_atoms = FakeAtomGroup([atoms[3]], universe=universe)
    atoms[3].bonded_atoms = FakeAtomGroup([atoms[2]], universe=universe)

    residues = [FakeResidue(FakeAtomGroup([atom], universe=universe)) for atom in atoms]

    data_container = MagicMock()
    data_container.residues = residues

    result = dd.method_res_bonds(data_container, "residue")

    assert len(result) == 1
    assert result[0].indices == [0, 1, 2, 3]
