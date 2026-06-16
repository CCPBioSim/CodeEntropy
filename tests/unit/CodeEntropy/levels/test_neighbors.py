"""Unit tests for neighbour-count and symmetry helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from CodeEntropy.levels.neighbors import Chem, Neighbors


class FakeSearch:
    """Minimal Search test double."""

    def __init__(self):
        self.rad_calls = []
        self.grid_calls = []

    def get_RAD_neighbors(self, *, universe, mol_id, frame_source, frame_index):
        self.rad_calls.append(
            {
                "universe": universe,
                "mol_id": mol_id,
                "frame_source": frame_source,
                "frame_index": frame_index,
            }
        )
        return [1, 2]

    def get_grid_neighbors(
        self,
        *,
        universe,
        mol_id,
        highest_level,
        frame_source,
        frame_index,
    ):
        self.grid_calls.append(
            {
                "universe": universe,
                "mol_id": mol_id,
                "highest_level": highest_level,
                "frame_source": frame_source,
                "frame_index": frame_index,
            }
        )
        return [1]


class FakeRdkitMol:
    """Minimal RDKit-like molecule."""

    def __init__(self, *, heavy_atoms: int, total_atoms: int):
        self._heavy_atoms = heavy_atoms
        self._total_atoms = total_atoms

    def GetNumHeavyAtoms(self):
        return self._heavy_atoms

    def GetNumAtoms(self):
        return self._total_atoms


class FakeAtomSelection:
    """Minimal AtomGroup-like object returned by select_atoms."""

    def __init__(self, *, length: int, rdkit_mol=None):
        self._length = length
        self._rdkit_mol = rdkit_mol
        self.convert_to = MagicMock(return_value=rdkit_mol)

    def __len__(self):
        return self._length


class FakeMolecule:
    """Minimal molecule fragment for RDKit conversion tests."""

    def __init__(self, *, dummy_atoms, heavy_fragment, rdkit_mol):
        self._dummy_atoms = dummy_atoms
        self._heavy_fragment = heavy_fragment
        self._rdkit_mol = rdkit_mol
        self.convert_to = MagicMock(return_value=rdkit_mol)

    def select_atoms(self, selection):
        if selection == "prop mass < 0.1":
            return self._dummy_atoms
        if selection == "prop mass > 0.1":
            return self._heavy_fragment
        raise AssertionError(f"Unexpected selection: {selection}")


class FakeAtomsWithoutElements:
    """Universe atoms object without an elements attribute."""

    def __init__(self, fragments):
        self.fragments = fragments


class FakeAtomsWithElements:
    """Universe atoms object with an elements attribute."""

    def __init__(self, fragments):
        self.fragments = fragments
        self.elements = ["C"]


def test_neighbors_accepts_injected_search_dependency():
    search = FakeSearch()

    helper = Neighbors(search=search)

    assert helper._search is search


def test_get_frame_neighbor_counts_rad():
    helper = Neighbors()
    helper._search = FakeSearch()

    universe = object()
    frame_source = object()
    groups = {0: [0, 1]}
    levels = [["united_atom"], ["united_atom", "residue"]]

    result = helper.get_frame_neighbor_counts(
        universe=universe,
        levels=levels,
        groups=groups,
        frame_source=frame_source,
        frame_index=5,
        search_type="RAD",
    )

    assert result == {0: (4, 2)}
    assert helper._search.rad_calls == [
        {
            "universe": universe,
            "mol_id": 0,
            "frame_source": frame_source,
            "frame_index": 5,
        },
        {
            "universe": universe,
            "mol_id": 1,
            "frame_source": frame_source,
            "frame_index": 5,
        },
    ]


def test_get_frame_neighbor_counts_grid():
    helper = Neighbors()
    helper._search = FakeSearch()

    universe = object()
    frame_source = object()
    groups = {0: [0, 1]}
    levels = [["united_atom"], ["united_atom", "residue"]]

    result = helper.get_frame_neighbor_counts(
        universe=universe,
        levels=levels,
        groups=groups,
        frame_source=frame_source,
        frame_index=3,
        search_type="grid",
    )

    assert result == {0: (2, 2)}
    assert helper._search.grid_calls == [
        {
            "universe": universe,
            "mol_id": 0,
            "highest_level": "united_atom",
            "frame_source": frame_source,
            "frame_index": 3,
        },
        {
            "universe": universe,
            "mol_id": 1,
            "highest_level": "united_atom",
            "frame_source": frame_source,
            "frame_index": 3,
        },
    ]


def test_get_frame_neighbor_counts_empty_group():
    helper = Neighbors()
    helper._search = FakeSearch()

    result = helper.get_frame_neighbor_counts(
        universe=object(),
        levels=[],
        groups={0: []},
        frame_source=object(),
        frame_index=0,
        search_type="RAD",
    )

    assert result == {0: (0, 0)}


def test_get_frame_neighbor_counts_converts_frame_index_and_handles_multiple_groups():
    search = FakeSearch()
    helper = Neighbors(search=search)

    universe = object()
    frame_source = object()
    groups = {
        7: [0],
        9: [1, 2],
    }
    levels = [
        ["united_atom"],
        ["united_atom", "residue"],
        ["united_atom", "residue", "polymer"],
    ]

    result = helper.get_frame_neighbor_counts(
        universe=universe,
        levels=levels,
        groups=groups,
        frame_source=frame_source,
        frame_index="12",
        search_type="RAD",
    )

    assert result == {
        7: (2, 1),
        9: (4, 2),
    }
    assert search.rad_calls == [
        {
            "universe": universe,
            "mol_id": 0,
            "frame_source": frame_source,
            "frame_index": 12,
        },
        {
            "universe": universe,
            "mol_id": 1,
            "frame_source": frame_source,
            "frame_index": 12,
        },
        {
            "universe": universe,
            "mol_id": 2,
            "frame_source": frame_source,
            "frame_index": 12,
        },
    ]


def test_get_frame_neighbor_counts_raises_for_unknown_search_type():
    helper = Neighbors()
    helper._search = FakeSearch()

    with pytest.raises(ValueError, match="unknown search_type"):
        helper.get_frame_neighbor_counts(
            universe=object(),
            levels=[["united_atom"]],
            groups={0: [0]},
            frame_source=object(),
            frame_index=0,
            search_type="unknown",
        )


def test_get_neighbors_for_molecule_rad_delegates_to_search():
    search = FakeSearch()
    helper = Neighbors(search=search)

    universe = object()
    frame_source = object()

    result = helper._get_neighbors_for_molecule(
        universe=universe,
        molecule_id=3,
        highest_level="residue",
        frame_source=frame_source,
        frame_index=8,
        search_type="RAD",
    )

    assert result == [1, 2]
    assert search.rad_calls == [
        {
            "universe": universe,
            "mol_id": 3,
            "frame_source": frame_source,
            "frame_index": 8,
        }
    ]


def test_get_neighbors_for_molecule_grid_delegates_to_search_with_highest_level():
    search = FakeSearch()
    helper = Neighbors(search=search)

    universe = object()
    frame_source = object()

    result = helper._get_neighbors_for_molecule(
        universe=universe,
        molecule_id=4,
        highest_level="polymer",
        frame_source=frame_source,
        frame_index=9,
        search_type="grid",
    )

    assert result == [1]
    assert search.grid_calls == [
        {
            "universe": universe,
            "mol_id": 4,
            "highest_level": "polymer",
            "frame_source": frame_source,
            "frame_index": 9,
        }
    ]


def test_get_neighbors_for_molecule_raises_for_unknown_search_type():
    helper = Neighbors(search=FakeSearch())

    with pytest.raises(ValueError, match="unknown search_type unknown"):
        helper._get_neighbors_for_molecule(
            universe=object(),
            molecule_id=0,
            highest_level="united_atom",
            frame_source=object(),
            frame_index=0,
            search_type="unknown",
        )


def test_get_symmetry_calls_helpers_for_first_molecule_in_each_group():
    helper = Neighbors()
    calls = []

    def fake_get_rdkit_mol(universe, molecule_id):
        calls.append(molecule_id)
        return f"mol-{molecule_id}", 2 + molecule_id, molecule_id

    helper._get_rdkit_mol = fake_get_rdkit_mol
    helper._get_symmetry_number = lambda rdkit_mol, number_heavy, number_hydrogen: (
        number_heavy + number_hydrogen
    )
    helper._get_linear = lambda rdkit_mol, number_heavy: number_heavy == 2

    symmetry, linear = helper.get_symmetry(
        universe=object(),
        groups={7: [0, 1], 9: [2]},
    )

    assert calls == [0, 2]
    assert symmetry == {7: 2, 9: 6}
    assert linear == {7: True, 9: False}


def test_get_symmetry_returns_zero_for_empty_groups():
    helper = Neighbors()
    helper._get_rdkit_mol = MagicMock()

    symmetry, linear = helper.get_symmetry(universe=object(), groups={7: []})

    assert symmetry == {7: 0}
    assert linear == {7: False}
    helper._get_rdkit_mol.assert_not_called()


def test_get_symmetry_propagates_error_from_rdkit_conversion():
    helper = Neighbors()
    helper._get_rdkit_mol = MagicMock(side_effect=RuntimeError("bad molecule"))

    with pytest.raises(RuntimeError, match="bad molecule"):
        helper.get_symmetry(universe=object(), groups={7: [0]})


def test_get_rdkit_mol_guesses_elements_when_missing_and_uses_normal_conversion():
    rdkit_mol = FakeRdkitMol(heavy_atoms=2, total_atoms=6)

    dummy_atoms = FakeAtomSelection(length=0)
    heavy_fragment = FakeAtomSelection(length=2, rdkit_mol=rdkit_mol)
    molecule = FakeMolecule(
        dummy_atoms=dummy_atoms,
        heavy_fragment=heavy_fragment,
        rdkit_mol=rdkit_mol,
    )

    universe = SimpleNamespace(
        atoms=FakeAtomsWithoutElements([molecule]),
        guess_TopologyAttrs=MagicMock(),
    )

    out_mol, number_heavy, number_hydrogen = Neighbors._get_rdkit_mol(universe, 0)

    universe.guess_TopologyAttrs.assert_called_once_with(to_guess=["elements"])
    molecule.convert_to.assert_called_once_with("RDKIT", force=True)

    assert out_mol is rdkit_mol
    assert number_heavy == 2
    assert number_hydrogen == 4


def test_get_rdkit_mol_skips_guessing_when_elements_exist():
    rdkit_mol = FakeRdkitMol(heavy_atoms=1, total_atoms=4)

    molecule = FakeMolecule(
        dummy_atoms=FakeAtomSelection(length=0),
        heavy_fragment=FakeAtomSelection(length=1, rdkit_mol=rdkit_mol),
        rdkit_mol=rdkit_mol,
    )

    universe = SimpleNamespace(
        atoms=FakeAtomsWithElements([molecule]),
        guess_TopologyAttrs=MagicMock(),
    )

    out_mol, number_heavy, number_hydrogen = Neighbors._get_rdkit_mol(universe, 0)

    universe.guess_TopologyAttrs.assert_not_called()
    molecule.convert_to.assert_called_once_with("RDKIT", force=True)

    assert out_mol is rdkit_mol
    assert number_heavy == 1
    assert number_hydrogen == 3


def test_get_rdkit_mol_uses_heavy_fragment_when_dummy_atoms_are_present():
    rdkit_mol = FakeRdkitMol(heavy_atoms=3, total_atoms=8)

    dummy_atoms = FakeAtomSelection(length=2)
    heavy_fragment = FakeAtomSelection(length=3, rdkit_mol=rdkit_mol)
    molecule = FakeMolecule(
        dummy_atoms=dummy_atoms,
        heavy_fragment=heavy_fragment,
        rdkit_mol=MagicMock(),
    )

    universe = SimpleNamespace(
        atoms=FakeAtomsWithElements([molecule]),
        guess_TopologyAttrs=MagicMock(),
    )

    out_mol, number_heavy, number_hydrogen = Neighbors._get_rdkit_mol(universe, 0)

    molecule.convert_to.assert_not_called()
    heavy_fragment.convert_to.assert_called_once_with(
        "RDKIT",
        force=True,
        inferrer=None,
    )

    assert out_mol is rdkit_mol
    assert number_heavy == 3
    assert number_hydrogen == 5


def test_get_rdkit_mol_falls_back_to_inferrer_none_when_normal_conversion_fails():
    rdkit_mol = FakeRdkitMol(heavy_atoms=2, total_atoms=5)

    molecule = FakeMolecule(
        dummy_atoms=FakeAtomSelection(length=0),
        heavy_fragment=FakeAtomSelection(length=2, rdkit_mol=rdkit_mol),
        rdkit_mol=rdkit_mol,
    )
    molecule.convert_to = MagicMock(
        side_effect=[RuntimeError("bad valence"), rdkit_mol]
    )

    universe = SimpleNamespace(
        atoms=FakeAtomsWithElements([molecule]),
        guess_TopologyAttrs=MagicMock(),
    )

    out_mol, number_heavy, number_hydrogen = Neighbors._get_rdkit_mol(universe, 0)

    assert molecule.convert_to.call_args_list == [
        call("RDKIT", force=True),
        call("RDKIT", force=True, inferrer=None),
    ]

    assert out_mol is rdkit_mol
    assert number_heavy == 2
    assert number_hydrogen == 3


def test_get_symmetry_number_uses_heavy_atom_matches_for_multi_heavy_molecule():
    helper = Neighbors()
    rdkit_mol = MagicMock()
    rdkit_mol.GetSubstructMatches.return_value = [1, 2, 3]

    with patch("CodeEntropy.levels.neighbors.Chem.RemoveHs", return_value="heavy"):
        assert helper._get_symmetry_number(rdkit_mol, 2, 0) == 3

    rdkit_mol.GetSubstructMatches.assert_called_once_with(
        "heavy",
        uniquify=False,
        useChirality=True,
    )


def test_get_symmetry_number_uses_full_molecule_for_single_heavy_with_hydrogens():
    helper = Neighbors()
    rdkit_mol = MagicMock()
    rdkit_mol.GetSubstructMatches.return_value = [1, 2]

    assert helper._get_symmetry_number(rdkit_mol, 1, 4) == 2

    rdkit_mol.GetSubstructMatches.assert_called_once_with(
        rdkit_mol,
        uniquify=False,
        useChirality=True,
    )


def test_get_symmetry_number_returns_zero_for_single_heavy_without_hydrogens():
    assert Neighbors()._get_symmetry_number(MagicMock(), 1, 0) == 0


def test_get_linear_for_one_or_two_heavy_atoms():
    helper = Neighbors()

    assert helper._get_linear(MagicMock(), 1) is False
    assert helper._get_linear(MagicMock(), 2) is True


def test_get_linear_for_larger_molecule_uses_sp_hybridisation_count():
    helper = Neighbors()
    rdkit_mol = MagicMock()

    sp_atom = MagicMock()
    sp_atom.GetHybridization.return_value = Chem.HybridizationType.SP
    non_sp_atom = MagicMock()
    non_sp_atom.GetHybridization.return_value = Chem.HybridizationType.SP3

    rdkit_heavy = MagicMock()
    rdkit_heavy.GetAtoms.return_value = [sp_atom, sp_atom, non_sp_atom]

    with patch("CodeEntropy.levels.neighbors.Chem.RemoveHs", return_value=rdkit_heavy):
        assert helper._get_linear(rdkit_mol, 4) is True


def test_get_linear_for_larger_molecule_returns_false_when_too_few_sp_atoms():
    helper = Neighbors()
    rdkit_mol = MagicMock()

    non_sp_atom = MagicMock()
    non_sp_atom.GetHybridization.return_value = Chem.HybridizationType.SP3

    rdkit_heavy = MagicMock()
    rdkit_heavy.GetAtoms.return_value = [non_sp_atom, non_sp_atom, non_sp_atom]

    with patch("CodeEntropy.levels.neighbors.Chem.RemoveHs", return_value=rdkit_heavy):
        assert helper._get_linear(rdkit_mol, 4) is False
