# import os
import logging
import sys

# import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np

logger = logging.getLogger(__name__)


# from ast import arg


# from numpy import linalg as la

# import math

# from CodeEntropy import CustomFunctions as CF
# from CodeEntropy import GeometricFunctions as GF
# from CodeEntropy import UnitsAndConversions as UAC
# from CodeEntropy import LevelFunctions as LF
# from CodeEntropy import Utils
# from CodeEntropy import Writer
# import multiprocessing as mp
# from functools import partial


# import pandas as pd
# from MDAnalysis.analysis import distances

np.set_printoptions(threshold=sys.maxsize)


def get_neighbours(molecule_i, reduced_atom):
    """
    Molecules in the coordination shell are obtained using the relative angular distance
    (RAD) algorithm, as presented in J.Higham and R. H. Henchman, J. Chem. Phys., 2016,
    145, 084108.

    Input
    -----
      molecule: DataContainer type with the information on the molecule of interest for
      which are finding the neighbours
      reduced_atom: MDAnalysis universe object containing the system of interest
    Returns
    -------
      neighbours_dict :  dictionary - dictionary of neighbours for the molecule - should
      contain the type of neighbour molecule and the number of neighbour
      molecules of that species
      neighbours_array: array containing all neighbouring molecules
    """
    number_molecules = len(reduced_atom.atoms.fragments)
    neighbours_dict = {}
    neighbours_array = []
    # go through molecules to see if they're in the coordination shell of the molecule
    # of interest
    for j in range(0, number_molecules):
        molecule_j = reduced_atom.atoms.fragments[j]
        # to avoid adding a molecule as its own neighbour
        if molecule_i != molecule_j:
            # we have already found a neighbour
            if len(neighbours_array) > 0:
                for molecule_k in (
                    neighbours_array
                    # check if every closer molecule that is already in the neighbour
                    # list is unblocked
                ):
                    # distance between the molecule of interest and the molecule
                    # that we check if it's a neighbour
                    r_ij = mda.analysis.distances.dist(molecule_i, molecule_j)
                    # distance between the molecule of interest and the neighbours
                    # already identified
                    r_ik = mda.analysis.distances.dist(molecule_i, molecule_k)
                    r_jk = mda.analysis.distances.dist(molecule_j, molecule_k)
                    cos_theta_jik = (r_ij**2 + r_ik**2 - r_jk**2) / (
                        2 * r_ij * r_ik
                    )  # cosine of the angle subintended at i by j and k
                    if (r_ij) ** (-2) < (r_ik) ** (-2) * cos_theta_jik:
                        break  # j is not a neighbour as it is blocked by k
                    else:
                        neighbours_array.append(molecule_j.atoms.resids[0])
                        if molecule_j.smth in neighbours_dict.keys():
                            neighbours_dict[molecule_j.atoms.resnames[0]] += 1
                        else:
                            neighbours_dict[molecule_j.atoms.resnames[0]] = 1
            else:  # first neighbour
                # r_ij = mda.analysis.distances.dist(molecule_i, molecule_j)
                # print(r_ij)
                neighbours_array.append(molecule_j.atoms.resids[0])
                neighbours_dict[molecule_j.atoms.resnames[0]] = 1

    logger.debug(f"Neighbours dictionary: {neighbours_dict}")
    logger.debug(f"Neighbours array: {neighbours_array}")

    return neighbours_dict, neighbours_array
