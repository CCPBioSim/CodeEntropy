import contextlib
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from CodeEntropy.levels.neighbors import Neighbors


class _FakeProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return 1

    def advance(self, *args, **kwargs):
        return None


@contextlib.contextmanager
def _fake_progress_bar(*_args, **_kwargs):
    yield _FakeProgress()


def test_raises_error_unknown_search_type():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0]}
    n_frames = 2
    search_type = "weird"

    with pytest.raises(ValueError):
        neighbors.get_neighbors(universe, levels, groups, n_frames, search_type)


def test_average_number_neighbors_RAD():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0]}
    n_frames = 2
    search_type = "RAD"

    neighbors._search.get_RAD_neighbors = MagicMock(side_effect=[[1, 2, 3], [1, 3]])

    result = neighbors.get_neighbors(universe, levels, groups, n_frames, search_type)

    assert result == {0: np.float64(2.5)}


def test_average_number_neighbors_grid():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0]}
    n_frames = 2
    search_type = "grid"

    neighbors._search.get_grid_neighbors = MagicMock(side_effect=[[1, 2, 3], [1, 3]])

    result = neighbors.get_neighbors(universe, levels, groups, n_frames, search_type)

    assert result == {0: np.float64(2.5)}


def test_average_number_neighbors_RAD_multiple():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0, 1]}
    n_frames = 2
    search_type = "RAD"

    neighbors._search.get_RAD_neighbors = MagicMock(
        side_effect=[[1, 2, 3, 5], [1, 3], [2, 3, 4, 5], [3, 5]]
    )

    result = neighbors.get_neighbors(universe, levels, groups, n_frames, search_type)

    assert result == {0: np.float64(3.0)}


def test_get_symmetry_number_res():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    number_heavy = 3
    number_hydrogen = 8

    rdkit_mol.GetSubstructMatches = MagicMock(
        side_effect=[((0, 1, 2), (0, 2, 1), (1, 0, 2))]
    )

    class _FakeRDKit_Chem:
        """Class to mock rdkit functionality."""

        def RemoveHs(mol):
            rdkit_heavy = MagicMock()
            return rdkit_heavy

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_symmetry_number(
            rdkit_mol, number_heavy, number_hydrogen
        )

    assert result == 3


def test_get_symmetry_number_ua():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    number_heavy = 1
    number_hydrogen = 2

    rdkit_mol.GetSubstructMatches = MagicMock(side_effect=[((0, 1, 2), (0, 2, 1))])

    result = neighbors._get_symmetry_number(rdkit_mol, number_heavy, number_hydrogen)

    assert result == 2


def test_get_symmetry_number_sphere():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    number_heavy = 1
    number_hydrogen = 0

    rdkit_mol.GetSubstructMatches = MagicMock(side_effect=[((0, 1, 2), (0, 2, 1))])

    result = neighbors._get_symmetry_number(rdkit_mol, number_heavy, number_hydrogen)

    assert result == 0


def test_get_linear_ua():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    number_heavy = 1

    class _FakeRDKit_Chem:
        """Class to mock rdkit functionality."""

        def RemoveHs(mol):
            rdkit_heavy = MagicMock()
            return rdkit_heavy

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_linear(rdkit_mol, number_heavy)

    assert not result


def test_get_linear_diatomic():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    number_heavy = 2

    class _FakeRDKit_Chem:
        """Class to mock rdkit functionality."""

        def RemoveHs(mol):
            rdkit_heavy = MagicMock()
            return rdkit_heavy

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_linear(rdkit_mol, number_heavy)

    assert result


def test_get_linear_true():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    rdkit_heavy = MagicMock()
    number_heavy = 3
    a1 = MagicMock()
    a2 = MagicMock()
    a3 = MagicMock()

    class _FakeRDKit_Chem:
        """Class to mock rdkit functionality."""

        @staticmethod
        def RemoveHs(mol):
            return rdkit_heavy

        class HybridizationType:
            SP = "SP"

    rdkit_heavy.GetAtoms = MagicMock(return_value=[a1, a2, a3])

    a1.GetHybridization = MagicMock(return_value="SP3")
    a2.GetHybridization = MagicMock(return_value="SP")
    a3.GetHybridization = MagicMock(return_value="SP3")

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_linear(rdkit_mol, number_heavy)

    assert result is True


def test_get_linear_false():
    neighbors = Neighbors()
    rdkit_mol = MagicMock()
    rdkit_heavy = MagicMock()
    number_heavy = 3
    a1 = MagicMock()
    a2 = MagicMock()
    a3 = MagicMock()

    class _FakeRDKit_Chem:
        """Class to mock rdkit functionality."""

        @staticmethod
        def RemoveHs(mol):
            return rdkit_heavy

        class HybridizationType:
            SP = "SP"

    rdkit_heavy.GetAtoms = MagicMock(return_value=[a1, a2, a3])
    a1.GetHybridization = MagicMock(return_value="SP3")
    a2.GetHybridization = MagicMock(return_value="SP3")
    a3.GetHybridization = MagicMock(return_value="SP3")

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_linear(rdkit_mol, number_heavy)

    assert result is False


def test_get_symmetry_returns_dicts_for_single_group():
    neighbors = Neighbors()
    universe = MagicMock()
    groups = {7: [42, 99]}

    rdkit_mol = MagicMock()

    neighbors._get_rdkit_mol = MagicMock(return_value=(rdkit_mol, 5, 8))
    neighbors._get_symmetry_number = MagicMock(return_value=12)
    neighbors._get_linear = MagicMock(return_value=True)

    symmetry_number, linear = neighbors.get_symmetry(universe, groups)

    assert symmetry_number == {7: 12}
    assert linear == {7: True}

    neighbors._get_rdkit_mol.assert_called_once_with(universe, 42)
    neighbors._get_symmetry_number.assert_called_once_with(rdkit_mol, 5, 8)
    neighbors._get_linear.assert_called_once_with(rdkit_mol, 5)


def test_get_symmetry_uses_first_molecule_in_each_group_only():
    neighbors = Neighbors()
    universe = MagicMock()
    groups = {
        0: [10, 11, 12],
        1: [20, 21],
    }

    rdkit_mol_0 = MagicMock()
    rdkit_mol_1 = MagicMock()

    neighbors._get_rdkit_mol = MagicMock(
        side_effect=[
            (rdkit_mol_0, 3, 6),
            (rdkit_mol_1, 4, 8),
        ]
    )
    neighbors._get_symmetry_number = MagicMock(side_effect=[2, 4])
    neighbors._get_linear = MagicMock(side_effect=[False, True])

    symmetry_number, linear = neighbors.get_symmetry(universe, groups)

    assert symmetry_number == {0: 2, 1: 4}
    assert linear == {0: False, 1: True}

    assert neighbors._get_rdkit_mol.call_args_list == [
        call(universe, 10),
        call(universe, 20),
    ]


def test_get_symmetry_calls_helpers_for_each_group_in_order():
    neighbors = Neighbors()
    universe = MagicMock()
    groups = {
        3: [100],
        5: [200],
    }

    rdkit_mol_a = MagicMock()
    rdkit_mol_b = MagicMock()

    neighbors._get_rdkit_mol = MagicMock(
        side_effect=[
            (rdkit_mol_a, 1, 2),
            (rdkit_mol_b, 7, 0),
        ]
    )
    neighbors._get_symmetry_number = MagicMock(side_effect=[9, 1])
    neighbors._get_linear = MagicMock(side_effect=[True, False])

    symmetry_number, linear = neighbors.get_symmetry(universe, groups)

    assert symmetry_number == {3: 9, 5: 1}
    assert linear == {3: True, 5: False}

    assert neighbors._get_symmetry_number.call_args_list == [
        call(rdkit_mol_a, 1, 2),
        call(rdkit_mol_b, 7, 0),
    ]
    assert neighbors._get_linear.call_args_list == [
        call(rdkit_mol_a, 1),
        call(rdkit_mol_b, 7),
    ]


def test_get_symmetry_returns_empty_dicts_for_empty_groups():
    neighbors = Neighbors()
    universe = MagicMock()
    groups = {}

    neighbors._get_rdkit_mol = MagicMock()
    neighbors._get_symmetry_number = MagicMock()
    neighbors._get_linear = MagicMock()

    symmetry_number, linear = neighbors.get_symmetry(universe, groups)

    assert symmetry_number == {}
    assert linear == {}

    neighbors._get_rdkit_mol.assert_not_called()
    neighbors._get_symmetry_number.assert_not_called()
    neighbors._get_linear.assert_not_called()


def test_get_symmetry_propagates_error_from_get_rdkit_mol():
    neighbors = Neighbors()
    universe = MagicMock()
    groups = {0: [123]}

    neighbors._get_rdkit_mol = MagicMock(side_effect=RuntimeError("bad molecule"))
    neighbors._get_symmetry_number = MagicMock()
    neighbors._get_linear = MagicMock()

    with pytest.raises(RuntimeError, match="bad molecule"):
        neighbors.get_symmetry(universe, groups)

    neighbors._get_symmetry_number.assert_not_called()
    neighbors._get_linear.assert_not_called()


def test_get_rdkit_mol_guesses_elements_when_missing():
    neighbors = Neighbors()

    universe = MagicMock()
    molecule = MagicMock()
    dummy = MagicMock()

    del universe.atoms.elements
    universe.atoms.fragments = [molecule]

    molecule.select_atoms.side_effect = [dummy]
    dummy.__len__.return_value = 0

    rdkit_mol = MagicMock()
    rdkit_mol.GetNumHeavyAtoms.return_value = 3
    rdkit_mol.GetNumAtoms.return_value = 8
    molecule.convert_to.return_value = rdkit_mol

    result = neighbors._get_rdkit_mol(universe, 0)

    universe.guess_TopologyAttrs.assert_called_once_with(to_guess=["elements"])
    molecule.convert_to.assert_called_once_with("RDKIT", force=True)
    assert result == (rdkit_mol, 3, 5)


def test_get_rdkit_mol_does_not_guess_elements_when_present():
    neighbors = Neighbors()

    universe = MagicMock()
    molecule = MagicMock()
    dummy = MagicMock()

    universe.atoms.elements = ["C", "H"]
    universe.atoms.fragments = [molecule]

    molecule.select_atoms.side_effect = [dummy]
    dummy.__len__.return_value = 0

    rdkit_mol = MagicMock()
    rdkit_mol.GetNumHeavyAtoms.return_value = 2
    rdkit_mol.GetNumAtoms.return_value = 6
    molecule.convert_to.return_value = rdkit_mol

    result = neighbors._get_rdkit_mol(universe, 0)

    universe.guess_TopologyAttrs.assert_not_called()
    molecule.convert_to.assert_called_once_with("RDKIT", force=True)
    assert result == (rdkit_mol, 2, 4)


def test_get_rdkit_mol_uses_full_molecule_when_no_dummy_atoms():
    neighbors = Neighbors()

    universe = MagicMock()
    molecule = MagicMock()
    dummy = MagicMock()

    universe.atoms.elements = ["C"]
    universe.atoms.fragments = [molecule]

    molecule.select_atoms.side_effect = [dummy]
    dummy.__len__.return_value = 0

    rdkit_mol = MagicMock()
    rdkit_mol.GetNumHeavyAtoms.return_value = 4
    rdkit_mol.GetNumAtoms.return_value = 10
    molecule.convert_to.return_value = rdkit_mol

    result = neighbors._get_rdkit_mol(universe, 0)

    molecule.select_atoms.assert_called_once_with("prop mass < 0.1")
    molecule.convert_to.assert_called_once_with("RDKIT", force=True)
    assert result == (rdkit_mol, 4, 6)


def test_get_rdkit_mol_removes_dummy_atoms_and_uses_inferrer_none():
    neighbors = Neighbors()

    universe = MagicMock()
    molecule = MagicMock()
    dummy = MagicMock()
    frag = MagicMock()

    universe.atoms.elements = ["C"]
    universe.atoms.fragments = [molecule]

    molecule.select_atoms.side_effect = [dummy, frag]
    dummy.__len__.return_value = 2

    rdkit_mol = MagicMock()
    rdkit_mol.GetNumHeavyAtoms.return_value = 5
    rdkit_mol.GetNumAtoms.return_value = 12
    frag.convert_to.return_value = rdkit_mol

    result = neighbors._get_rdkit_mol(universe, 0)

    assert molecule.select_atoms.call_args_list == [
        (("prop mass < 0.1",),),
        (("prop mass > 0.1",),),
    ]
    frag.convert_to.assert_called_once_with("RDKIT", force=True, inferrer=None)
    molecule.convert_to.assert_not_called()
    assert result == (rdkit_mol, 5, 7)


def test_get_rdkit_mol_returns_correct_heavy_and_hydrogen_counts():
    neighbors = Neighbors()

    universe = MagicMock()
    molecule = MagicMock()
    dummy = MagicMock()

    universe.atoms.elements = ["O", "H", "H"]
    universe.atoms.fragments = [molecule]

    molecule.select_atoms.side_effect = [dummy]
    dummy.__len__.return_value = 0

    rdkit_mol = MagicMock()
    rdkit_mol.GetNumHeavyAtoms.return_value = 1
    rdkit_mol.GetNumAtoms.return_value = 3
    molecule.convert_to.return_value = rdkit_mol

    rdkit_out, number_heavy, number_hydrogen = neighbors._get_rdkit_mol(universe, 0)

    assert rdkit_out is rdkit_mol
    assert number_heavy == 1
    assert number_hydrogen == 2
