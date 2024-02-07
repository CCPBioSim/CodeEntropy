#!/usr/bin/env python

"""
Functions for common trigonometric calculations
"""

import MDAnalysis
import numpy as np


def get_neighbourlist(atom, neighbours, dimensions, max_cutoff=9e9):
    """
    Use MDAnalysis to get distances between an atom and neighbours within
    a given cutoff. Each atom index pair sorted by distance are outputted.

    :param atom: (3,) array of an atom coordinates.
    :param neighbours: MDAnalysis array of heavy atoms in the system,
        not the atom itself and not bonded to the atom.
    :param dimensions: (6,) array of system box dimensions.
    """
    pairs, distances = MDAnalysis.lib.distances.capped_distance(
        atom,
        neighbours.positions,
        max_cutoff=max_cutoff,
        min_cutoff=None,
        box=dimensions,
        method=None,
        return_distances=True,
    )
    neighbour_indices = neighbours[pairs[:][:, 1]].indices
    sorted_distances, sorted_indices = zip(
        *sorted(zip(distances, neighbour_indices), key=lambda x: x[0])
    )
    return np.array(sorted_indices), np.array(sorted_distances)


def get_angle(a, b, c, dimensions):
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
    cosine_angle = (dist_ac**2 - dist_bc**2 - dist_ba**2) / (
        -2 * dist_bc * dist_ba
    )
    return cosine_angle


def get_distance(a, b, dimensions):
    """
    calculates distance and accounts for PBCs.

    :param a: (3,) array of atom cooordinates
    :param b: (3,) array of atom cooordinates
    :param dimensions: (3,) array of system box dimensions.
    :type a: numpy array
    :type b: numpy array
    """
    delta = np.abs(b - a)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    distance = np.sqrt((delta**2).sum(axis=-1))
    return distance


def get_vector(a, b, dimensions):
    """
    get vector of two coordinates over PBCs.

    :param a: (3,) array of atom cooordinates
    :param b: (3,) array of atom cooordinates
    :param dimensions: (3,) array of system box dimensions.
    :type a: numpy array
    :type b: numpy array
    """
    delta = b - a
    delta_wrapped = []
    for delt, box in zip(delta, dimensions):
        if delt < 0 and delt < 0.5 * box:
            delt = delt + box
        if delt > 0 and delt > 0.5 * box:
            delt = delt - box
        delta_wrapped.append(delt)
    delta_wrapped = np.array(delta_wrapped)

    return delta_wrapped
