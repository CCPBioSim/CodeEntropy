import contextlib
from unittest.mock import MagicMock, patch

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
    search_type = "weird"

    with pytest.raises(ValueError):
        neighbors.get_neighbors(universe, levels, groups, search_type)


def test_average_number_neighbors_RAD():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0]}
    search_type = "RAD"

    neighbors._search.get_RAD_neighbors = MagicMock(side_effect=[[1, 2, 3], [1, 3]])

    result = neighbors.get_neighbors(universe, levels, groups, search_type)

    assert result == {0: np.float64(2.5)}


def test_average_number_neighbors_grid():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0]}
    search_type = "grid"

    neighbors._search.get_grid_neighbors = MagicMock(side_effect=[[1, 2, 3], [1, 3]])

    result = neighbors.get_neighbors(universe, levels, groups, search_type)

    assert result == {0: np.float64(2.5)}


def test_average_number_neighbors_RAD_multiple():
    neighbors = Neighbors()

    universe = MagicMock()
    universe.trajectory.__len__.return_value = 2
    levels = {0: ["united_atom"]}
    groups = {0: [0, 1]}
    search_type = "RAD"

    neighbors._search.get_RAD_neighbors = MagicMock(
        side_effect=[[1, 2, 3, 5], [1, 3], [2, 3, 4, 5], [3, 5]]
    )

    result = neighbors.get_neighbors(universe, levels, groups, search_type)

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

        def RemoveHs(mol):
            rdkit_heavy = MagicMock()
            return rdkit_heavy

        class HybridizationType:
            def SP():
                return "SP"

    rdkit_heavy.GetAtoms = MagicMock(return_value=[a1, a2, a3])
    a1.GetHybridization = MagicMock(return_value="SP2")
    a2.GetHybridization = MagicMock(return_value="SP")
    a3.GetHybridization = MagicMock(return_value="SP3")

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_linear(rdkit_mol, number_heavy)

    assert result


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

        def RemoveHs(mol):
            rdkit_heavy = MagicMock()
            return rdkit_heavy

        class HybridizationType:
            def SP():
                return "SP"

    rdkit_heavy.GetAtoms = MagicMock(return_value=[a1, a2, a3])
    a1.GetHybridization = MagicMock(return_value="SP3")
    a2.GetHybridization = MagicMock(return_value="SP3")
    a3.GetHybridization = MagicMock(return_value="SP3")

    with patch("CodeEntropy.levels.neighbors.Chem", _FakeRDKit_Chem):
        result = neighbors._get_linear(rdkit_mol, number_heavy)

    assert not result
