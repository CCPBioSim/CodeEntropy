"""These functions find neighbors.

There are different functions for different types of neighbor searching.
Currently RAD is the default with grid as an alternative.
"""

import MDAnalysis as mda
import numpy as np


class Search:
    """
    Class for functions to find neighbors.
    """

    def __init__(self):
        """
        Initializes the Search class with a placeholder for the system
        trajectory.
        """
        self._universe = None
        self._mol_id = None

    def _get_fragment_coms(self, universe):
        """
        Precompute fragment centres of mass.

        Args:
            universe: MDAnalysis universe object.

        Returns:
            np.ndarray: Array of fragment COMs.
        """
        return np.array([frag.center_of_mass() for frag in universe.atoms.fragments])

    def _get_distances(self, coms, i_coords, dimensions):
        """
        Function to calculate distances between a central point and all COMs.
        Takes periodic boundary conditions into account.

        Args:
            coms: array of fragment COMs
            i_coords: coordinates of central molecule
            dimensions: simulation box dimensions

        Returns:
            np.ndarray: distances to all molecules
        """
        delta = coms - i_coords
        delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
        delta = np.where(delta < -0.5 * dimensions, delta + dimensions, delta)
        return np.sqrt((delta**2).sum(axis=1))

    def get_RAD_neighbors(self, universe, mol_id):
        """
        Find the neighbors of molecule with index mol_id.

        Args:
            universe: The MDAnalysis universe of the system.
            mol_id (int): the index for the central molecule.

        Returns:
            neighbor_indices (list of ints): the list of neighboring molecule
                indices.
        """
        number_molecules = len(universe.atoms.fragments)

        # Precompute COMs once
        coms = self._get_fragment_coms(universe)

        # Central molecule position
        central_position = coms[mol_id]

        # Compute all distances in one vectorised call
        distances_array = self._get_distances(
            coms, central_position, universe.dimensions[:3]
        )

        # Build distance dict excluding self
        distances = {}
        for molecule_index_j in range(number_molecules):
            if molecule_index_j != mol_id:
                distances[molecule_index_j] = distances_array[molecule_index_j]

        # Sort distances smallest to largest
        sorted_dist = sorted(distances.items(), key=lambda item: item[1])

        # Get indices of neighbors
        neighbor_indices = self._get_RAD_indices(
            central_position,
            sorted_dist,
            coms,
            universe.dimensions[:3],
            number_molecules,
        )

        return neighbor_indices

    def _get_RAD_indices(
        self, i_coords, sorted_distances, coms, dimensions, number_molecules
    ):
        # pylint: disable=too-many-locals
        r"""
        For a given set of atom coordinates, find its RAD shell from the distance
        sorted list, truncated to the closest 30 molecules.

        This function calculates coordination shells using RAD the relative
        angular distance, as defined first in DOI:10.1063/1.4961439
        where molecules are defined as neighbors if
        they fulfil the following condition:

        .. math::
            \Bigg(\frac{1}{r_{ij}}\Bigg)^2 >
            \Bigg(\frac{1}{r_{ik}}\Bigg)^2 \cos \theta_{jik}

        For a given particle :math:`i`, neighbor :math:`j` is in its coordination
        shell if :math:`k` is not blocking particle :math:`j`. In this implementation
        of RAD, we enforce symmetry, whereby neighboring particles must be in each
        others coordination shells.

        Args:
            i_coords: xyz centre of mass of molecule :math:`i`
            sorted_distances: list of index and distance pairs sorted by distance
            coms: precomputed center of mass array
            dimensions: system box dimensions
            number_molecules: total number of molecules

        Returns:
            shell: list of indices of particles in the RAD shell of neighbors.
        """
        shell = []
        count = -1
        limit = min(number_molecules - 1, 30)

        for y in range(limit):
            count += 1

            j_idx = sorted_distances[y][0]
            r_ij = sorted_distances[y][1]
            j_coords = coms[j_idx]

            blocked = False

            for z in range(count):
                k_idx = sorted_distances[z][0]
                r_ik = sorted_distances[z][1]
                k_coords = coms[k_idx]

                costheta_jik = self.get_angle(j_coords, i_coords, k_coords, dimensions)

                if np.isnan(costheta_jik):
                    break

                LHS = (1 / r_ij) ** 2
                RHS = ((1 / r_ik) ** 2) * costheta_jik

                if LHS < RHS:
                    blocked = True
                    break

            if not blocked:
                shell.append(j_idx)

        return shell

    def get_angle(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray, dimensions: np.ndarray
    ):
        """
        Get the angle between three atoms, taking into account periodic
        boundary conditions.

        b is the vertex of the angle.

        Args:
            a: (3,) array of atom coordinates
            b: (3,) array of atom coordinates
            c: (3,) array of atom coordinates
            dimensions: (3,) array of system box dimensions.

        Returns:
            cosine_angle: float, cosine of the angle abc.
        """
        ba = np.abs(a - b)
        bc = np.abs(c - b)
        ac = np.abs(c - a)

        ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
        bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
        ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)

        dist_ba = np.sqrt((ba**2).sum(axis=-1))
        dist_bc = np.sqrt((bc**2).sum(axis=-1))
        dist_ac = np.sqrt((ac**2).sum(axis=-1))

        cosine_angle = (dist_ac**2 - dist_bc**2 - dist_ba**2) / (-2 * dist_bc * dist_ba)

        return cosine_angle

    def get_grid_neighbors(self, universe, mol_id, highest_level):
        """
        Use MDAnalysis neighbor search to find neighbors.

        For molecules with just one united atom, use the "A" search level to
        find neighboring atoms. For larger molecules use the "R" search level
        to find neighboring residues.

        The atoms/residues of the molecule of interest are removed from the
        neighbor list.

        Args:
            universe: MDAnalysis universe object for system.
            mol_id: int, the index for the molecule of interest.
            highest_level: str, molecule level.

        Returns:
            neighbors: MDAnalysis atomgroup of the neighboring particles.
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
