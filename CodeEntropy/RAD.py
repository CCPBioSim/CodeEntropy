"""
These functions calculate coordination shells using RAD the relative
angular distance.
"""

import numpy as np
import waterEntropy.maths.trig as Trig
from waterEntropy.analysis.shells import ShellCollection


def get_RAD_shell(
    UA, system, shells: ShellCollection, sorted_indices=None, sorted_distances=None
):
    """
    For a given united atom, find its RAD shell, returning the atom indices
    for the heavy atoms that are in its shell.

    :param UA: mdanalysis instance of a united atom in a frame
    :param system: mdanalysis instance of atoms in a frame
    :param shells: ShellCollection instance
    """
    # 1. first check if a shell has already been found for this UA
    shell = shells.find_shell(UA.index)
    if not shell:
        # 2. get the nearest neighbours for the UA, sorted from closest to
        # furthest
        if sorted_indices is None:
            sorted_indices, sorted_distances = Trig.get_sorted_neighbours(
                UA.index, system
            )
        # 3. now find the RAD shell of the UA
        shell_indices = get_RAD_neighbours(
            UA.position, sorted_indices, sorted_distances, system
        )
        # 4. populate the class instance for RAD shells
        shells.add_data(UA.index, shell_indices)
        shell = shells.find_shell(UA.index)
    return shell


def get_RAD_neighbours(i_coords, sorted_indices, sorted_distances, system):
    # pylint: disable=too-many-locals
    r"""
    For a given set of atom coordinates, find its RAD shell from the distance
    sorted atom list, truncated to the closests 30 atoms.

    This function calculates coordination shells using RAD the relative
    angular distance, as defined first in DOI:10.1063/1.4961439
    where united atoms (heavy atom + bonded Hydrogens) are defined as neighbours if
    they fulfil the following condition:

    .. math::
        \Bigg(\frac{1}{r_{ij}}\Bigg)^2>\Bigg(\frac{1}{r_{ik}}\Bigg)^2 \cos \theta_{jik}

    For a given particle :math:`i`, neighbour :math:`j` is in its coordination
    shell if :math:`k` is not blocking particle :math:`j`. In this implementation
    of RAD, we enforce symmetry, whereby neighbouring particles must be in each
    others coordination shells.

    :param i_coords: xyz coordinates of atom :math:`i`
    :param sorted_indices: list of atom indices sorted from closest to
        furthest from atom :math:`i`
    :param sorted_distances: list of atom distances sorted from closest to
        furthest from atom :math:`i`
    :param system: mdanalysis instance of atoms in a frame
    """
    # 1. truncate neighbour list to closest 25 united atoms
    range_limit = min(len(sorted_distances), 25)
    shell = []
    count = -1
    # 2. iterate through neighbours from closest to furthest
    for y in sorted_indices[:range_limit]:
        count += 1
        y_idx = np.where(sorted_indices == y)[0][0]
        j = system.atoms.indices[y]
        j_coords = system.atoms.positions[y]
        rij = sorted_distances[y_idx]
        blocked = False
        # 3. iterate through neighbours than atom j and check if they block
        # it from atom i
        for z in sorted_indices[:count]:  # only closer UAs can block
            z_idx = np.where(sorted_indices == z)[0][0]
            k_coords = system.atoms.positions[z]
            rik = sorted_distances[z_idx]
            # 4. find the angle jik
            costheta_jik = Trig.get_angle(
                j_coords, i_coords, k_coords, system.dimensions[:3]
            )
            if np.isnan(costheta_jik):
                break
            # 5. check if k blocks j from i
            LHS = (1 / rij) ** 2
            RHS = ((1 / rik) ** 2) * costheta_jik
            if LHS < RHS:
                blocked = True
                break
        # 6. if j is not blocked from i by k, then its in i's shell
        if blocked is False:
            shell.append(j)
    return shell
