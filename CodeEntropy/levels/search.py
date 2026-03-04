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

        central_position = universe.atoms.fragments[mol_id].center_of_mass()

        # Find distances between molecule of interest and other molecules in the system
        distances = {}
        for molecule_index_j in range(number_molecules):
            if molecule_index_j != mol_id:
                j_position = universe.atoms.fragments[molecule_index_j].center_of_mass()
                distances[molecule_index_j] = self.get_distance(
                    j_position, central_position, universe.dimensions
                )

        # Sort distances smallest to largest
        sorted_dist = sorted(distances.items(), key=lambda item: item[1])

        # Get indices of neighbors
        neighbor_indices = self._get_RAD_indices(
            central_position, sorted_dist, universe, number_molecules
        )

        return neighbor_indices

    def _get_RAD_indices(self, i_coords, sorted_distances, system, number_molecules):
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
            sorted_indices: dict of index and distance pairs sorted by distance
            system: mdanalysis instance of atoms in a frame

        Returns:
            shell: list of indices of particles in the RAD shell of neighbors.
        """
        # 1. truncate neighbor list to closest 30 united atoms and iterate
        # through neighbors from closest to furthest/
        shell = []
        count = -1
        limit = min(number_molecules - 1, 30)
        for y in range(limit):
            count += 1
            j_idx = sorted_distances[y][0]
            j_coords = system.atoms.fragments[j_idx].center_of_mass()
            r_ij = sorted_distances[y][1]
            blocked = False
            # 3. iterate through neighbors other than atom j and check if they block
            # it from molecule i
            for z in range(count):  # only closer units can block
                k_idx = sorted_distances[z][0]
                k_coords = system.atoms.fragments[k_idx].center_of_mass()
                r_ik = sorted_distances[z][1]
                # 4. find the angle jik
                costheta_jik = self.get_angle(
                    j_coords, i_coords, k_coords, system.dimensions[:3]
                )
                if np.isnan(costheta_jik):
                    break
                # 5. check if k blocks j from i
                LHS = (1 / r_ij) ** 2
                RHS = ((1 / r_ik) ** 2) * costheta_jik
                if LHS < RHS:
                    blocked = True
                    break
            # 6. if j is not blocked from i by k, then its in i's shell
            if blocked is False:
                shell.append(j_idx)

        return shell

    def get_angle(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray, dimensions: np.ndarray
    ):
        """
        Get the angle between three atoms, taking into account periodic
        bondary conditions.

        b is the vertex of the angle.

        Pairwise differences between the coordinates are used with the
        distances calculated as the square root of the sum of the squared
        x, y, and z coordinates.

        Args:
            a: (3,) array of atom cooordinates
            b: (3,) array of atom cooordinates
            c: (3,) array of atom cooordinates
            dimensions: (3,) array of system box dimensions.

        Returns:
            cosine_angle: float, cosine of the angle abc.
        """
        # Differences in positions
        ba = np.abs(a - b)
        bc = np.abs(c - b)
        ac = np.abs(c - a)

        # Correct for periodic boundary conditions
        ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
        bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
        ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)

        # Get distances
        dist_ba = np.sqrt((ba**2).sum(axis=-1))
        dist_bc = np.sqrt((bc**2).sum(axis=-1))
        dist_ac = np.sqrt((ac**2).sum(axis=-1))

        # Trigonometry
        cosine_angle = (dist_ac**2 - dist_bc**2 - dist_ba**2) / (-2 * dist_bc * dist_ba)

        return cosine_angle

    def get_distance(self, j_position, i_position, dimensions):
        """
        Function to calculate the distance between two points.
        Take periodic boundary conditions into account.

        Args:
            j_position: the x, y, z coordinates of point 1
            i_position: the x, y, z coordinates of the other point
            dimensions: the dimensions of the simulation box

        Returns:
            distance: the distance between the two points
        """

        x = []
        total = 0
        for coord in range(3):
            x.append(j_position[coord] - i_position[coord])
            if x[coord] > 0.5 * dimensions[coord]:
                x[coord] = x[coord] - dimensions[coord]
            total += x[coord] ** 2
        distance = np.sqrt(total)

        return distance

    def get_grid_neighbors(self, universe, search_object, mol_id, highest_level):
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
            # For united atom size molecules, use the grid search
            # to find neighboring atoms
            search_level = "A"
            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                search_object,
                molecule_atom_group,
                radius=3.0,
                level=search_level,
            )
            # Make sure that the neighbors list does not include
            # atoms from the central molecule
            #  neighbors = search - fragment.residues
            neighbors = search - molecule_atom_group
        else:
            # For larger molecules, use the grid search to find neighboring residues
            search_level = "R"
            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                search_object,
                molecule_atom_group,
                radius=3.5,
                level=search_level,
            )
            # Make sure that the neighbors list does not include
            # residues from the central molecule
            neighbors = search - fragment.residues

        return neighbors
