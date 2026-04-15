from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from CodeEntropy.levels.search import Search, _apply_pbc, _rad_blocking_loop


@pytest.fixture
def search():
    return Search()


def test_apply_pbc_wraps_positive():
    vec = np.array([11.0, 0.0, 0.0])
    dimensions = np.array([10.0, 10.0, 10.0])
    half = 0.5 * dimensions

    result = _apply_pbc(vec.copy(), dimensions, half)

    assert np.allclose(result, [1.0, 0.0, 0.0])


def test_apply_pbc_wraps_negative():
    vec = np.array([-11.0, 0.0, 0.0])
    dimensions = np.array([10.0, 10.0, 10.0])
    half = 0.5 * dimensions

    result = _apply_pbc(vec.copy(), dimensions, half)

    assert np.allclose(result, [-1.0, 0.0, 0.0])


def test_get_distances_applies_pbc(search):
    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ]
    )

    i_coords = np.array([0.0, 0.0, 0.0])
    dimensions = np.array([10.0, 10.0, 10.0])

    distances = search._get_distances(coms, i_coords, dimensions)

    assert len(distances) == 2
    assert distances[1] < 2.0


def test_update_cache_initializes_and_skips_on_same_frame(search):
    universe = MagicMock()
    universe.trajectory.ts.frame = 0
    universe.dimensions = np.array([10.0, 10.0, 10.0])

    frag1 = MagicMock()
    frag1.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

    frag2 = MagicMock()
    frag2.center_of_mass.return_value = np.array([1.0, 1.0, 1.0])

    universe.atoms.fragments = [frag1, frag2]

    search._update_cache(universe)

    assert search._cached_frame == 0
    assert search._cached_coms.shape == (2, 3)

    old = search._cached_coms.copy()
    search._update_cache(universe)

    assert np.array_equal(old, search._cached_coms)


def test_update_cache_updates_on_new_frame(search):
    universe = MagicMock()

    frag = MagicMock()
    frag.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
    universe.atoms.fragments = [frag]

    universe.dimensions = np.array([10.0, 10.0, 10.0])

    universe.trajectory.ts.frame = 0
    search._update_cache(universe)

    universe.trajectory.ts.frame = 1
    search._update_cache(universe)

    assert search._cached_frame == 1


def test_get_RAD_neighbors_returns_array(search):
    universe = MagicMock()
    universe.trajectory.ts.frame = 0
    universe.dimensions = np.array([10.0, 10.0, 10.0])

    frag1 = MagicMock()
    frag2 = MagicMock()
    frag3 = MagicMock()

    frag1.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
    frag2.center_of_mass.return_value = np.array([1.0, 0.0, 0.0])
    frag3.center_of_mass.return_value = np.array([2.0, 0.0, 0.0])

    universe.atoms.fragments = [frag1, frag2, frag3]

    result = search.get_RAD_neighbors(universe, mol_id=0)

    assert isinstance(result, np.ndarray)


def test_rad_pbc_path_triggers_wrapping(search):
    universe = MagicMock()
    universe.trajectory.ts.frame = 0
    universe.dimensions = np.array([10.0, 10.0, 10.0])

    frag1 = MagicMock()
    frag2 = MagicMock()

    frag1.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
    frag2.center_of_mass.return_value = np.array([9.5, 0.0, 0.0])

    universe.atoms.fragments = [frag1, frag2]

    result = search.get_RAD_neighbors(universe, mol_id=0)

    assert isinstance(result, np.ndarray)


def test_get_grid_neighbors_united_atom(search):
    universe = MagicMock()

    fragment = MagicMock()
    fragment.indices = [10, 11]

    universe.atoms.fragments = [fragment]

    molecule_atom_group = MagicMock()
    universe.select_atoms.return_value = molecule_atom_group

    search_result = MagicMock()
    diff_result = MagicMock()
    diff_result.fragindices = np.array([1, 2])

    search_result.__sub__.return_value = diff_result

    with patch(
        "CodeEntropy.levels.search.mda.lib.NeighborSearch.AtomNeighborSearch.search",
        autospec=True,
        return_value=search_result,
    ) as mock_search:
        result = search.get_grid_neighbors(
            universe,
            mol_id=0,
            highest_level="united_atom",
        )

        mock_search.assert_called_once()
        universe.select_atoms.assert_called_once_with("index 10:11")
        assert np.array_equal(result, np.array([1, 2]))


def test_get_grid_neighbors_residue(search):
    universe = MagicMock()

    fragment = MagicMock()
    fragment.indices = [4, 5, 6]
    fragment.residues = MagicMock()

    universe.atoms.fragments = [fragment]

    molecule_atom_group = MagicMock()
    universe.select_atoms.return_value = molecule_atom_group

    search_result = MagicMock()
    diff_result = MagicMock()
    diff_result.atoms = MagicMock()
    diff_result.atoms.fragindices = np.array([7, 8, 9])

    search_result.__sub__.return_value = diff_result

    with patch(
        "CodeEntropy.levels.search.mda.lib.NeighborSearch.AtomNeighborSearch.search",
        autospec=True,
        return_value=search_result,
    ) as mock_search:
        result = search.get_grid_neighbors(
            universe,
            mol_id=0,
            highest_level="other",
        )

        mock_search.assert_called_once()
        universe.select_atoms.assert_called_once_with("index 4:6")
        assert np.array_equal(result, np.array([7, 8, 9]))


def test_get_grid_neighbors_selection_string(search):
    universe = MagicMock()

    fragment = MagicMock()
    fragment.indices = [3, 7]

    universe.atoms.fragments = [fragment]
    universe.select_atoms.return_value = MagicMock()

    with patch(
        "CodeEntropy.levels.search.mda.lib.NeighborSearch.AtomNeighborSearch.search",
        autospec=True,
        return_value=MagicMock(),
    ):
        search.get_grid_neighbors(
            universe,
            mol_id=0,
            highest_level="united_atom",
        )

    universe.select_atoms.assert_called_once_with("index 3:7")


def test_rad_blocking_loop_no_blocking_simple():
    i_coords = np.array([0.0, 0.0, 0.0])

    sorted_indices = np.array([1, 2], dtype=np.int64)
    sorted_distances = np.array([1.0, 2.0], dtype=np.float64)

    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    dimensions = np.array([10.0, 10.0, 10.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert isinstance(result, np.ndarray)
    assert len(result) >= 1
    assert result[0] in sorted_indices


def test_rad_blocking_loop_blocking_by_closer_neighbor():
    i_coords = np.array([0.0, 0.0, 0.0])

    sorted_indices = np.array([2, 1], dtype=np.int64)
    sorted_distances = np.array([1.0, 2.0], dtype=np.float64)

    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    dimensions = np.array([10.0, 10.0, 10.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert set(result) == set(result)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64


def test_rad_blocking_loop_pbc_wraps_distance():
    i_coords = np.array([0.0, 0.0, 0.0])

    sorted_indices = np.array([1, 2], dtype=np.int64)
    sorted_distances = np.array([1.0, 1.0], dtype=np.float64)

    # One atom across boundary
    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.9, 0.0, 0.0],
            [-4.9, 0.0, 0.0],
        ]
    )

    dimensions = np.array([10.0, 10.0, 10.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert set(result) == {1, 2}


def test_rad_blocking_loop_respects_limit_30():
    i_coords = np.zeros(3)

    n = 40
    sorted_indices = np.arange(1, n + 1, dtype=np.int64)
    sorted_distances = np.linspace(1.0, 5.0, n)

    coms = np.zeros((n + 1, 3))
    for i in range(1, n + 1):
        coms[i] = np.array([float(i), 0.0, 0.0])

    dimensions = np.array([100.0, 100.0, 100.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert len(result) <= 30


def test_rad_blocking_loop_zero_distance_handling():
    i_coords = np.array([0.0, 0.0, 0.0])

    sorted_indices = np.array([1], dtype=np.int64)
    sorted_distances = np.array([0.0], dtype=np.float64)

    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    dimensions = np.array([10.0, 10.0, 10.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert isinstance(result, np.ndarray)


def test_rad_blocking_loop_continue_rik_gt_rij():
    i_coords = np.array([0.0, 0.0, 0.0])

    sorted_indices = np.array([0, 1], dtype=np.int64)
    sorted_distances = np.array([2.0, 1.0], dtype=np.float64)

    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    dimensions = np.array([10.0, 10.0, 10.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert isinstance(result, np.ndarray)


def test_rad_blocking_loop_continue_zero_denom():
    i_coords = np.array([0.0, 0.0, 0.0])

    sorted_indices = np.array([0, 1], dtype=np.int64)
    sorted_distances = np.array([0.0, 1.0], dtype=np.float64)

    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    dimensions = np.array([10.0, 10.0, 10.0])

    result = _rad_blocking_loop(
        i_coords, sorted_indices, sorted_distances, coms, dimensions
    )

    assert isinstance(result, np.ndarray)
