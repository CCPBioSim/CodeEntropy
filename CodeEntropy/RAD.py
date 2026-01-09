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
            x = j_position[0] - central_position[0]
            y = j_position[1] - central_position[1]
            z = j_position[2] - central_position[2]
            distances[molecule_index_j] = np.sqrt(x**2 + y**2 + z**2)

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


# def get_neighbourlist(
#    atom: np.ndarray, neighbours, dimensions: np.ndarray, max_cutoff=9e9
# ):
#    """
#    Use MDAnalysis to get distances between an atom and neighbours within
#    a given cutoff. Each atom index pair sorted by distance are outputted.
#
#    :param atom: (3,) array of an atom coordinates.
#    :param neighbours: MDAnalysis array of heavy atoms in the system,
#        not the atom itself and not bonded to the atom.
#    :param dimensions: (6,) array of system box dimensions.
#    :param max_cutoff: set the maximum cutoff value for finding neighbour distances
#    """
#    # check atom coords are not in neighbour coords list
#    if not (atom == neighbours.positions).all(axis=1).any():
#        pairs, distances = MDAnalysis.lib.distances.capped_distance(
#            atom,
#            neighbours.positions,
#            max_cutoff=max_cutoff,
#            min_cutoff=None,
#            box=dimensions,
#            method=None,
#            return_distances=True,
#        )
#        neighbour_indices = neighbours[pairs[:][:, 1]].indices
#        sorted_distances, sorted_indices = zip(
#            *sorted(zip(distances, neighbour_indices), key=lambda x: x[0])
#        )
#        return np.array(sorted_indices), np.array(sorted_distances)
#    raise ValueError(
#        f"Atom coordinates {atom} in neighbour list {neighbours.positions[:10]}"
#    )


# def get_sorted_neighbours(i_idx: int, system, max_cutoff=10):
#    """
#    For a given atom, find neighbouring united atoms from closest to furthest
#    within a given cutoff.
#
#    :param i_idx: idx of atom i
#    :param system: mdanalysis instance of atoms in a frame
#    :param max_cutoff: set the maximum cutoff value for finding neighbours
#    """
#    i_coords = system.atoms.positions[i_idx]
#    # 1. get the heavy atom neighbour distances within a given distance cutoff
#    # CHECK Find out which of the options below is better for RAD shells
#    #       Should the central atom bonded UAs be allowed to block?
#    #       This was not done in original code, keep the same here
#    neighbours = system.select_atoms(
#        f"""mass 2 to 999 and not index {i_idx}
#                                    and not bonded index {i_idx}"""
#        # f"""mass 2 to 999 and not index {i_idx}"""  # bonded UAs can block
#    )
#    # 2. Get the neighbours sorted from closest to furthest
#    sorted_indices, sorted_distances = get_neighbourlist(
#        i_coords, neighbours.atoms, system.dimensions, max_cutoff
#    )
#    return sorted_indices, sorted_distances


# def get_shell_neighbour_selection(
#    shell, donator, system, heavy_atoms=True, max_cutoff=10
# ):
#    """
#    get shell neighbours ordered by ascending distance, this is used for
#    finding possible hydrogen bonding neighbours.
#
#    :param shell: the instance for class waterEntropy.neighbours.RAD.RAD
#        containing coordination shell neighbours
#    :param donator: the mdanalysis object for the donator
#    :param system: mdanalysis instance of all atoms in current frame
#    :param heavy_atoms: consider heavy atoms in a shell as neighbours
#    :max_cutoff: set the maximum cutoff value for finding neighbours
#    """
#    # 1a. Select heavy atoms in shell, can only donate to heavy atoms in the shell
#    neighbours = Selections.get_selection(system, "index", shell.UA_shell)
#    if not heavy_atoms:
#        # 1b. Select all atoms in the shell, included bonded to atoms (Hs included)
#        #   Can donate to any atoms in a shell
#        all_shell_bonded = neighbours[:].bonds.indices
#        all_shell_indices = list(set().union(*all_shell_bonded))
#        # can donate to any atom in the shell
#        neighbours = Selections.get_selection(system, "index", all_shell_indices)
#    # 1c. can donate to any neighbours outside a shell, not used
#    # neighbours = system.select_atoms(
#    f"all and not index {donator.index} and not bonded index {donator.index}")
#    # 2. Get the neighbours sorted from closest to furthest
#    sorted_indices, sorted_distances = get_neighbourlist(
#        donator.position, neighbours, system.dimensions, max_cutoff
#    )
#
#    return sorted_indices, sorted_distances


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
