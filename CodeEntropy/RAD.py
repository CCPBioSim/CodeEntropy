"""
These functions calculate coordination shells using RAD the relative
angular distance.
"""

import numpy as np


def get_RAD_neighbors(universe, mol_id):
    """
    Find the neighbors of molecule with index mol_id.
    """
    number_molecules = len(universe.atoms.fragments)

    central_position = universe.atoms.fragments[mol_id].center_of_mass()

    # Find distances between molecule of interest and other molecules in the system
    distances = {}
    for molecule_index_j in range(number_molecules):
        if molecule_index_j != mol_id:
            j_position = universe.atoms.fragments[molecule_index_j].center_of_mass()
            distances[molecule_index_j] = get_distance(
                j_position, central_position, universe.dimensions
            )

    # Sort distances smallest to largest
    sorted_dist = sorted(distances.items(), key=lambda item: item[1])

    # Get indices of neighbors
    neighbor_indices = get_RAD_indices(central_position, sorted_dist, universe)

    return neighbor_indices


def get_RAD_indices(i_coords, sorted_distances, system):
    # pylint: disable=too-many-locals
    r"""
    For a given set of atom coordinates, find its RAD shell from the distance
    sorted list, truncated to the closest 30 molecules.

    This function calculates coordination shells using RAD the relative
    angular distance, as defined first in DOI:10.1063/1.4961439
    where molecules are defined as neighbours if
    they fulfil the following condition:

    .. math::
        \Bigg(\frac{1}{r_{ij}}\Bigg)^2>\Bigg(\frac{1}{r_{ik}}\Bigg)^2 \cos \theta_{jik}

    For a given particle :math:`i`, neighbour :math:`j` is in its coordination
    shell if :math:`k` is not blocking particle :math:`j`. In this implementation
    of RAD, we enforce symmetry, whereby neighbouring particles must be in each
    others coordination shells.

    :param i_coords: xyz centre of mass of molecule :math:`i`
    :param sorted_indices: dict of index and distance pairs sorted by distance
    :param system: mdanalysis instance of atoms in a frame
    """
    # 1. truncate neighbour list to closest 30 united atoms
    shell = []
    count = -1
    # 2. iterate through neighbours from closest to furthest
    for y in range(30):
        count += 1
        j_idx = sorted_distances[y][0]
        j_coords = system.atoms.fragments[j_idx].center_of_mass()
        r_ij = sorted_distances[y][1]
        blocked = False
        # 3. iterate through neighbours other than atom j and check if they block
        # it from molecule i
        for z in range(count):  # only closer units can block
            k_idx = sorted_distances[z][0]
            k_coords = system.atoms.fragments[k_idx].center_of_mass()
            r_ik = sorted_distances[z][1]
            # 4. find the angle jik
            costheta_jik = get_angle(
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


def get_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, dimensions: np.ndarray):
    """
    Get the angle between three atoms, taking into account PBC.

    :param a: (3,) array of atom cooordinates
    :param b: (3,) array of atom cooordinates
    :param c: (3,) array of atom cooordinates
    :param dimensions: (3,) array of system box dimensions.
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


def get_distance(j_position, i_position, dimensions):
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
