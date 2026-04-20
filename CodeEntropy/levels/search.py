"""These functions find neighbors.

There are different functions for different types of neighbor searching.
Currently RAD is the default with grid as an alternative.
"""

import MDAnalysis as mda
import numpy as np
from numba import njit


@njit
def _apply_pbc(vec, dimensions, half_dimensions):
    """
    Apply minimum image convention for periodic boundary conditions.

    Args:
        vec (np.ndarray):
            Vector to wrap.
        dimensions (np.ndarray):
            Simulation box dimensions.
        half_dimensions (np.ndarray):
            Half box lengths.

    Returns:
        np.ndarray:
            Wrapped vector.
    """
    for d in range(3):
        if vec[d] > half_dimensions[d]:
            vec[d] -= dimensions[d]
        elif vec[d] < -half_dimensions[d]:
            vec[d] += dimensions[d]
    return vec


@njit
def _rad_blocking_loop(i_coords, sorted_indices, sorted_distances, coms, dimensions):
    """
    Perform RAD neighbor selection using a blocking criterion.

    This is a Numba-compiled implementation of the RAD algorithm, which
    determines whether a molecule j is a neighbor of molecule i by checking
    whether any closer molecule k blocks j based on angular and distance
    relationships.

    The criterion is based on:

        (1 / r_ij)^2 > (1 / r_ik)^2 * cos(theta_jik)

    where k blocks j if the inequality holds.

    Args:
        i_coords (np.ndarray):
            Coordinates of the central molecule.
        sorted_indices (np.ndarray):
            Indices of molecules sorted by distance from i.
        sorted_distances (np.ndarray):
            Distances corresponding to sorted_indices.
        coms (np.ndarray):
            Precomputed center of mass coordinates for all molecules.
        dimensions (np.ndarray):
            Simulation box dimensions for periodic boundary conditions.

    Returns:
        np.ndarray:
            Indices of molecules that belong to the RAD neighbor shell.
    """
    n = sorted_indices.shape[0]
    limit = min(n, 30)

    half_dimensions = 0.5 * dimensions

    inv_r2 = 1.0 / (sorted_distances * sorted_distances)

    shell = np.empty(limit, dtype=np.int64)
    count = 0

    for y in range(limit):
        j_idx = sorted_indices[y]
        r_ij = sorted_distances[y]
        j_coords = coms[j_idx]

        ba = j_coords - i_coords
        ba = _apply_pbc(ba, dimensions, half_dimensions)

        blocked = False

        for z in range(y):
            k_idx = sorted_indices[z]
            r_ik = sorted_distances[z]

            if r_ik > r_ij:
                continue

            k_coords = coms[k_idx]

            ac = k_coords - j_coords
            ac = _apply_pbc(ac, dimensions, half_dimensions)

            dist_ac2 = (ac * ac).sum()

            denom = -2.0 * r_ik * r_ij
            if denom == 0.0:
                continue

            costheta = (dist_ac2 - r_ik * r_ik - r_ij * r_ij) / denom

            if inv_r2[y] < inv_r2[z] * costheta:
                blocked = True
                break

        if not blocked:
            shell[count] = j_idx
            count += 1

    return shell[:count]


class Search:
    """
    Class for functions to find neighbors.
    """

    def __init__(self):
        """
        Initialize the Search class.

        This class includes frame-safe caching of fragment COMs and
        system dimensions to avoid recomputation while preserving
        identical results to the original implementation.
        """
        self._cached_frame = None
        self._cached_fragments = None
        self._cached_coms = None
        self._cached_dimensions = None

    def _update_cache(self, universe):
        """
        Update cached MDAnalysis data if the simulation frame has changed.

        Args:
            universe (MDAnalysis.Universe):
                MDAnalysis universe object containing the system.
        """
        current_frame = universe.trajectory.ts.frame

        if self._cached_frame == current_frame:
            return

        fragments = universe.atoms.fragments

        coms = np.array([frag.center_of_mass() for frag in fragments])

        self._cached_fragments = fragments
        self._cached_coms = coms
        self._cached_dimensions = universe.dimensions[:3]
        self._cached_frame = current_frame

    def _get_distances(self, coms, i_coords, dimensions):
        """
        Compute distances between a central coordinate and all fragment COMs
        using periodic boundary conditions.

        Args:
            coms (np.ndarray):
                Array of fragment center of mass coordinates.
            i_coords (np.ndarray):
                Coordinates of the reference (central) molecule.
            dimensions (np.ndarray):
                Simulation box dimensions.

        Returns:
            np.ndarray:
                Distances from the central molecule to all fragments.
        """
        delta = coms - i_coords

        half_dimensions = 0.5 * dimensions

        for d in range(3):
            delta[:, d] = np.where(
                delta[:, d] > half_dimensions[d],
                delta[:, d] - dimensions[d],
                delta[:, d],
            )
            delta[:, d] = np.where(
                delta[:, d] < -half_dimensions[d],
                delta[:, d] + dimensions[d],
                delta[:, d],
            )

        return np.sqrt((delta * delta).sum(axis=1))

    def get_RAD_neighbors(self, universe, mol_id, timestep):
        """
        Find RAD neighbors of a given molecule.

        Args:
            universe (MDAnalysis.Universe):
                The MDAnalysis universe of the system.
            mol_id (int):
                Index of the central molecule.

        Returns:
            np.ndarray:
                Indices of neighboring molecules identified via the RAD method.
        """
        universe.trajectory[timestep]
        self._update_cache(universe)

        fragments = self._cached_fragments
        coms = self._cached_coms
        dimensions = self._cached_dimensions

        number_molecules = len(fragments)

        central_position = coms[mol_id]

        distances_array = self._get_distances(coms, central_position, dimensions)

        indices = np.arange(number_molecules)

        mask = indices != mol_id
        filtered_indices = indices[mask]
        filtered_distances = distances_array[mask]

        order = np.argsort(filtered_distances, kind="mergesort")

        sorted_indices = filtered_indices[order]
        sorted_distances = filtered_distances[order]

        neighbor_indices = _rad_blocking_loop(
            central_position,
            sorted_indices,
            sorted_distances,
            coms,
            dimensions,
        )

        return neighbor_indices

    def get_grid_neighbors(self, universe, mol_id, highest_level, timestep):
        """
        Find neighbors using MDAnalysis grid-based neighbor search.

        For small molecules (united_atom), atom-level search is used.
        For larger molecules, residue-level search is used.

        Args:
            universe (MDAnalysis.Universe):
                MDAnalysis universe object for the system.
            mol_id (int):
                Index of the molecule of interest.
            highest_level (str):
                Molecule level ("united_atom" or other).

        Returns:
            np.ndarray:
                Fragment indices of neighboring molecules.
        """
        universe.trajectory[timestep]
        fragments = universe.atoms.fragments
        fragment = fragments[mol_id]

        search_object = mda.lib.NeighborSearch.AtomNeighborSearch(universe.atoms)

        selection_string = f"index {fragment.indices[0]}:{fragment.indices[-1]}"
        molecule_atom_group = universe.select_atoms(selection_string)

        if highest_level == "united_atom":
            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                search_object,
                molecule_atom_group,
                radius=3.0,
                level="A",
            )
            neighbors = search - molecule_atom_group
        else:
            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                search_object,
                molecule_atom_group,
                radius=3.5,
                level="R",
            )
            neighbors = search - fragment.residues
            neighbors = neighbors.atoms

        return neighbors.fragindices
