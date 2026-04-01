"""These functions find neighbors.

There are different functions for different types of neighbor searching.
Currently RAD is the default with grid as an alternative.
"""

import MDAnalysis as mda
import numpy as np
from numba import njit


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
        list[int]:
            Indices of molecules that belong to the RAD neighbor shell.
    """
    shell = []

    n = sorted_indices.shape[0]
    limit = min(n, 30)

    for y in range(limit):
        j_idx = sorted_indices[y]
        r_ij = sorted_distances[y]
        j_coords = coms[j_idx]

        blocked = False

        for z in range(y):
            k_idx = sorted_indices[z]
            r_ik = sorted_distances[z]
            k_coords = coms[k_idx]

            # Compute coordinate differences
            ba = np.abs(j_coords - i_coords)
            bc = np.abs(k_coords - i_coords)
            ac = np.abs(k_coords - j_coords)

            # Apply periodic boundary conditions
            ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
            bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
            ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)

            # Compute distances
            dist_ba = np.sqrt((ba**2).sum())
            dist_bc = np.sqrt((bc**2).sum())
            dist_ac = np.sqrt((ac**2).sum())

            # Cosine of angle jik
            costheta = (dist_ac**2 - dist_bc**2 - dist_ba**2) / (-2 * dist_bc * dist_ba)

            if np.isnan(costheta):
                break

            LHS = (1.0 / r_ij) ** 2
            RHS = ((1.0 / r_ik) ** 2) * costheta

            if LHS < RHS:
                blocked = True
                break

        if not blocked:
            shell.append(j_idx)

    return shell


class Search:
    """
    Class for functions to find neighbors.
    """

    def __init__(self):
        """
        Initialize the Search class.

        This class currently serves as a container for neighbor search
        methods operating on an MDAnalysis universe.
        """
        self._universe = None
        self._mol_id = None

    def _get_fragment_coms(self, universe):
        """
        Precompute center of mass for each molecular fragment.

        Args:
            universe (MDAnalysis.Universe):
                MDAnalysis universe object containing the system.

        Returns:
            np.ndarray:
                Array of shape (n_fragments, 3) containing COM coordinates.
        """
        return np.array([frag.center_of_mass() for frag in universe.atoms.fragments])

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
        delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
        delta = np.where(delta < -0.5 * dimensions, delta + dimensions, delta)
        return np.sqrt((delta**2).sum(axis=1))

    def get_RAD_neighbors(self, universe, mol_id):
        """
        Find RAD neighbors of a given molecule.

        Args:
            universe (MDAnalysis.Universe):
                The MDAnalysis universe of the system.
            mol_id (int):
                Index of the central molecule.

        Returns:
            list[int]:
                Indices of neighboring molecules identified via the RAD method.
        """
        number_molecules = len(universe.atoms.fragments)

        # Precompute COMs
        coms = self._get_fragment_coms(universe)

        # Central molecule position
        central_position = coms[mol_id]

        # Compute distances
        distances_array = self._get_distances(
            coms, central_position, universe.dimensions[:3]
        )

        # Prepare indices
        indices = np.arange(number_molecules)

        # Remove self
        mask = indices != mol_id
        filtered_indices = indices[mask]
        filtered_distances = distances_array[mask]

        # Sort by distance
        order = np.argsort(filtered_distances)

        sorted_indices = filtered_indices[order]
        sorted_distances = filtered_distances[order]

        # RAD blocking (Numba)
        neighbor_indices = _rad_blocking_loop(
            central_position,
            sorted_indices,
            sorted_distances,
            coms,
            universe.dimensions[:3],
        )

        return neighbor_indices

    def get_grid_neighbors(self, universe, mol_id, highest_level):
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
            list[int]:
                Fragment indices of neighboring molecules.
        """
        search_object = mda.lib.NeighborSearch.AtomNeighborSearch(universe.atoms)
        fragment = universe.atoms.fragments[mol_id]

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
