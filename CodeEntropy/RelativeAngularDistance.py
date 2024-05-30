#!/usr/bin/env python

"""
These functions calculate coordination shells using RAD the relative
angular distance, as defined first in DOI:10.1063/1.4961439
where united atoms (heavy atom + bonded Hydrogens) are defined as neighbours if
they fulfil the following condition:

.. math::
    \\Bigg(\frac{1}{r_{ij}}\\Bigg)^2>\\Bigg(\frac{1}{r_{ik}}\\Bigg)^2 \\cos \theta_{jik}

For a given particle i, neighbour j is in its coordination shell if k is not
blocking particle j. In this implementation of RAD, we enforce symmetry, whereby
neighbouring particles must be in each others coordination shells.
"""

import numpy as np

from CodeEntropy.utils.selections import find_molecule_UAs, get_selection
from CodeEntropy.utils.trig import get_angle, get_neighbourlist


class RAD:  # pylint: disable=too-few-public-methods
    """
    Class for coordination shells defined using the RAD implementation
    The shells dictionary contains the central atom_idx as the key and
    the instance of the RAD class as the value.
    """

    shells = {}  # save shell instances in here

    def __init__(self, atom_idx: int, UA_shell: list[int]):
        self.atom_idx = atom_idx
        self.UA_shell = UA_shell
        self.nearest_nonlike_idx = None
        self.labels = None
        self.donates_to_labels = None
        self.accepts_from_labels = None
        RAD.shells[self.atom_idx] = self  # add instance to shell dict

    @classmethod  # access to the class, but not to the instance
    def find_shell(cls, atom_idx: int):
        """
        Get the shell instance if atom_idx is a key in shells dictionary

        :param cls: instance of RAD class
        :param atom_idx: index of central atom in shell
        """
        return RAD.shells.get(atom_idx, None)


def find_interfacial_solvent(solutes, system):
    """
    For a given set of solute molecules, find the RAD shells for each UA in the
    molecules, if a solvent molecule is in the RAD shell, then save the solvent
    atom index to a list. A solvent is defined as molecule that constitutes a
    single UA.

    :param solutes: mdanalysis instance of a selection of atoms in solute
        molecules that are greater than one UA
    :param system: mdanalysis instance of atoms in a frame
    """
    solvent_indices = []
    molecules = solutes.fragments  # fragments is mdanalysis equiv to molecules
    for molecule in molecules:
        # find heavy atoms in the molecule
        UAs = find_molecule_UAs(molecule)
        for UA in UAs:
            shell = get_RAD_shell(UA, system)  # get the molecule UA shell
            RAD(UA.index, shell)  # add shell to class
            # for each neighbour in the RAD shell, find single UA molecules
            for n in shell:
                n_UA = get_selection(system, "index", [n])
                UAs = find_molecule_UAs(n_UA.fragments[0])  # 1 length list
                if len(UAs) == 1:  # single UA molecule is a solvent
                    if n_UA.indices[0] not in solvent_indices:
                        solvent_indices.append(n_UA.indices[0])
    return solvent_indices


def get_RAD_shell(UA, system):
    """
    For a given united atom, find its RAD shell, returning the atom indices
    for the heavy atoms that are in its shell.

    :param UA: mdanalysis instance of a united atom in a frame
    :param system: mdanalysis instance of atoms in a frame
    """
    # first check if a shell has already been found for this UA
    shell = RAD.find_shell(UA.index)
    if not shell:
        # get the nearest neighbours for the UA, sorted from closest to
        # furthest
        sorted_indices, sorted_distances = get_sorted_neighbours(UA.index, system)
        # now find the RAD shell
        shell = get_RAD_neighbours(
            UA.position, sorted_indices, sorted_distances, system
        )
        # populate the class instance for RAD shells
        RAD(UA.index, shell)
    else:
        shell = shell.UA_shell
    return shell


def get_sorted_neighbours(i_idx: int, system):
    """
    For a given atom, find the united atoms in its coordination shell. The
    shell instance from the RAD class is created for the given atom

    :param i_idx: idx of atom i
    :param RADshell: instance for RAD class
    :param system: mdanalysis instance of atoms in a frame
    """
    i_coords = system.atoms.positions[i_idx]
    # get the heavy atom neighbour distances within a given distance cutoff
    # CHECK Find out which of the options below is better for RAD shells
    #       Should the central atom bonded UAs be allowed to block?
    #       I would think yes, but this was not done in original code
    neighbours = system.select_atoms(
        # f"""mass 2 to 999 and not index {i_idx}
        #                             and not bonded index {i_idx}"""
        f"""mass 2 to 999 and not index {i_idx}"""  # bonded UAs can block
    )
    sorted_indices, sorted_distances = get_neighbourlist(
        i_coords, neighbours.atoms, system.dimensions, max_cutoff=25
    )
    return sorted_indices, sorted_distances


def get_RAD_neighbours(i_coords, sorted_indices, sorted_distances, system):
    # pylint: disable=too-many-locals
    """
    For a given set of atom coordinates, find its RAD shell from the distance
    sorted atom list, truncated to the closests 30 atoms.

    :param i_coords: xyz coordinates of an atom
    :param sorted_indices: list of atom indices sorted from closest to
        furthest from atom i
    :param sorted_distances: list of atom distances sorted from closest to
        furthest from atom i
    :param system: mdanalysis instance of atoms in a frame
    """
    # truncate neighbour list to closest 30 united atoms
    range_limit = min(len(sorted_distances), 30)
    shell = []
    count = -1
    # iterate through neighbours from closest to furthest
    for y in sorted_indices[:range_limit]:
        count += 1
        y_idx = np.where(sorted_indices == y)[0][0]
        j = system.atoms.indices[y]
        j_coords = system.atoms.positions[y]
        rij = sorted_distances[y_idx]
        blocked = False
        # iterate through neighbours than atom j and check if they block
        # it from atom i
        for z in sorted_indices[:count]:  # only closer UAs can block
            z_idx = np.where(sorted_indices == z)[0][0]
            k_coords = system.atoms.positions[z]
            rik = sorted_distances[z_idx]
            # find the angle jik
            costheta_jik = get_angle(
                j_coords, i_coords, k_coords, system.dimensions[:3]
            )
            if np.isnan(costheta_jik):
                break
            # check if k blocks j from i
            LHS = (1 / rij) ** 2
            RHS = ((1 / rik) ** 2) * costheta_jik
            if LHS < RHS:
                blocked = True
                break
        # if j is not blocked from i by k, then its in i's shell
        if blocked is False:
            shell.append(j)
    return shell


def get_shell_labels(atom_idx, system, shell):
    """
    Get the shell labels
    For a central UA, rank its coordination shell by proximity to that
    central UA's nearest non-like molecule UA.
    #_RESNAME = RAD shell from same molecule type, when nearest nonlike
        resid is the same as the reference.
    'X_RESNAME' = when same molecule type has different nearest nonlike resid.
    'RESNAME' = when molecule of different type is in RAD shell.
    '0_RESNAME' = closest different type molecule in RAD shell.
            (the one its assigned to, its nearest non-like!)

    :param atom_idx: atom index of central atom in coordination shell
    :param system: mdanalysis instance of atoms in a frame
    """
    center = system.atoms[atom_idx]
    # find the closest different UA in a shell
    nearest_nonlike_idx = get_nearest_nonlike(shell, system)
    # only find labels if a solute is in the shell
    if nearest_nonlike_idx is not None:
        shell_labels = []
        for n in shell.UA_shell:
            neighbour = system.atoms[n]
            # label nearest nonlike atom as "0_RESNAME"
            if n == nearest_nonlike_idx:
                shell_labels.append(f"0_{neighbour.resname}")
            # label other nonlike atoms as "RESNAME"
            if n != nearest_nonlike_idx and neighbour.resname != center.resname:
                shell_labels.append(neighbour.resname)
            # find RAD shells for shell constituents with same resname
            # as central atom
            if n != nearest_nonlike_idx and neighbour.resname == center.resname:
                neighbour_shell = RAD.find_shell(n)
                if not neighbour_shell:
                    neighbour_shell = get_RAD_shell(neighbour, system)
                    neighbour_shell = RAD(n, neighbour_shell)
                # find nearest nonlike of neighbours with same resname
                # as central atom
                neighbour_nearest_nonlike_idx = get_nearest_nonlike(
                    neighbour_shell, system
                )
                # if neighbour has a pure shell, then it is in the second
                # shell of the nearest nonlike
                if neighbour_nearest_nonlike_idx is None:
                    shell_labels.append(f"2_{center.resname}")
                else:
                    # if neighbours nearest nonlike is the same atom as
                    # central atom, assume it is in the first shell
                    if neighbour_nearest_nonlike_idx == nearest_nonlike_idx:
                        shell_labels.append(f"1_{center.resname}")
                    else:
                        # if neighbours nearest nonlike is not the same atom
                        # as central atom,  it is in the first shell
                        # of a different atom and labelled as "X_RESNAME"
                        shell_labels.append(f"X_{center.resname}")
        shell.labels = shell_labels  # sorted(shell_labels) #don't sort yet
        shell.nearest_nonlike_idx = nearest_nonlike_idx
    return shell


def get_nearest_nonlike(shell, system):
    """
    For a given shell, find the closest neighbour that is not the same
    atom/molecule type as the central united atom.

    :param shell: python instance for coordination shell
    :param system: mdanalysis instance of atoms in a frame
    """
    nearest_nonlike_idx = None
    center = system.atoms[shell.atom_idx]
    for n in shell.UA_shell:
        neighbour = system.atoms[n]
        if neighbour.resname != center.resname and neighbour.type != center.type:
            nearest_nonlike_idx = n
            break
    return nearest_nonlike_idx
